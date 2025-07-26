"""Main taxi mobility simulator."""

import gc
import json
import logging
import random

import matplotlib

matplotlib.use("Agg")
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import smopy
from matplotlib.animation import FuncAnimation
from PIL import Image

from ..config.settings import SimulationSettings
from ..data.reward_calculator import RewardCalculator, dist_on_sphere
from ..data.ride_probability import RideProbabilityCalculator
from ..models.car import Car
from ..utils.output import OutputManager
from ..utils.visualization import Visualizer

logger = logging.getLogger(__name__)


class TaxiSimulator:
    """Main taxi mobility simulator class."""

    def __init__(self, settings: SimulationSettings):
        """Initialize the simulator with given settings."""
        self.settings = settings
        self.reward_calculator = RewardCalculator()
        self.prob_calculator = RideProbabilityCalculator(self.reward_calculator)
        self.visualizer = Visualizer()
        self.output_manager = OutputManager(
            self.settings.get_output_dir(), self.visualizer
        )

        # Simulation state
        self.cars_list: list[Car] = []
        self.road_graph: nx.DiGraph = nx.DiGraph()
        self.edges_cars_dict: dict[tuple[int, int], list[Car]] = {}
        self.ride_probabilities: list[list[list[float]]] = []
        self.reward_areas: list[list[list[dict]]] = []
        self.total_rewards: list[float] = []
        self.car_trajectories: list[tuple] = []

        # Animation components
        self.fig = None
        self.ax = None
        self.line = None
        self.ride_line = None
        self.dest_line = None
        self.dest_ride_line = None
        self.title_text = None

        # Simulation metrics
        self.animation_count = 0
        self.current_time_slot = 0

        logger.info(f"Initialized simulator with {settings.number_of_cars} cars")

    def run(self) -> None:
        """Run the complete simulation."""
        try:
            logger.info("Starting taxi mobility simulation")

            # Setup phase
            self._setup_simulation()

            # Run simulation
            self._run_simulation()

            # Save results
            self._save_results()

            logger.info("Simulation completed successfully")

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

    def _setup_simulation(self) -> None:
        """Set up all simulation components."""
        logger.info("Setting up simulation components")

        # Load map and boundaries
        self.smopy_map, self.map_bounds = self._load_map()

        # Calculate area dimensions
        self._calculate_area_dimensions()

        # Load ride probabilities and rewards
        self.ride_probabilities, self.reward_areas = (
            self.prob_calculator.calculate_ride_probabilities(
                self.settings.network_file,
                self.settings.taxi_data_dir,
                self.settings.num_of_division,
            )
        )

        # Create road network
        self._create_road_network()

        # Initialize cars
        self._initialize_cars()

        # Setup animation
        if self.settings.save_animation:
            self._setup_animation()

        logger.info("Simulation setup completed")

    def _load_map(self) -> tuple[smopy.Map, tuple[float, float, float, float]]:
        """Load map data and create smopy map."""
        logger.info("Loading map data")

        # Read GeoJSON boundaries
        with self.settings.geojson_file.open() as f:
            geojson_data = json.load(f)

        # Extract boundaries
        coordinates = geojson_data["features"][0]["geometry"]["coordinates"][0]
        lons, lats = zip(*coordinates, strict=False)

        bounds = (min(lats), min(lons), max(lats), max(lons))
        logger.info(f"Map bounds: {bounds}")

        # Create smopy map
        smopy_map = smopy.Map(
            bounds,
            tileserver=self.settings.tile_server,
            tilesize=256,
            maxtiles=16,
            z=self.settings.map_zoom_level,
        )

        # Calculate pixel bounds
        px_min_lon, px_min_lat = smopy_map.to_pixels(bounds[0], bounds[1])
        px_max_lon, px_max_lat = smopy_map.to_pixels(bounds[2], bounds[3])

        map_bounds = (
            min(px_max_lon, px_min_lon),
            max(px_max_lon, px_min_lon),
            min(px_max_lat, px_min_lat),
            max(px_max_lat, px_min_lat),
        )

        # Save map image
        map_image_path = self.settings.get_output_dir() / "map.png"
        smopy_map.save_png(str(map_image_path))

        return smopy_map, map_bounds

    def _calculate_area_dimensions(self) -> None:
        """Calculate physical dimensions of area divisions."""
        # Get geographic bounds from map
        bounds = self.smopy_map.box

        # Calculate area dimensions in kilometers
        self.area_width_km = (
            dist_on_sphere(bounds[0], bounds[1], bounds[0], bounds[3])
            / self.settings.num_of_division
        )

        self.area_height_km = (
            dist_on_sphere(bounds[0], bounds[1], bounds[2], bounds[1])
            / self.settings.num_of_division
        )

        logger.info(
            f"Area dimensions: {self.area_width_km:.2f}km x {self.area_height_km:.2f}km"
        )

    def _create_road_network(self) -> None:
        """Create road network from SUMO XML file."""
        logger.info("Creating road network")

        # Parse XML
        tree = ET.parse(self.settings.network_file)
        root = tree.getroot()

        # Get boundaries
        conv_boundary, orig_boundary = self._get_boundaries(root)

        # Create network
        self.road_graph, self.node_mappings = self._build_network_graph(
            root, conv_boundary, orig_boundary
        )

        # Initialize edges_cars_dict
        for edge in self.road_graph.edges():
            self.edges_cars_dict[edge] = []

        logger.info(
            f"Created road network with {self.road_graph.number_of_nodes()} nodes and {self.road_graph.number_of_edges()} edges"
        )

    def _get_boundaries(self, root) -> tuple[list[float], list[float]]:
        """Extract boundaries from SUMO XML."""
        for child in root:
            if child.tag == "location":
                conv_boundary = list(
                    map(float, child.attrib["convBoundary"].split(","))
                )
                orig_boundary = list(
                    map(float, child.attrib["origBoundary"].split(","))
                )
                return conv_boundary, orig_boundary

        raise ValueError("No location tag found in network file")

    def _build_network_graph(
        self, root, conv_boundary: list[float], orig_boundary: list[float]
    ) -> tuple[nx.DiGraph, dict]:
        """Build NetworkX graph from SUMO XML data."""
        graph = nx.DiGraph()
        node_mappings = {
            "xy_to_id": {},
            "id_to_index": {},
            "index_to_id": {},
            "id_to_coordinate": {},
        }

        node_id = 0

        # Calculate area divisions
        top, bottom = orig_boundary[3], orig_boundary[1]
        leftmost, rightmost = orig_boundary[0], orig_boundary[2]
        x_division_size = abs(leftmost - rightmost) / self.settings.num_of_division
        y_division_size = abs(top - bottom) / self.settings.num_of_division

        for child in root:
            if child.tag == "edge" and not self._is_non_roadway(child):
                for lane_child in child:
                    if "shape" not in lane_child.attrib:
                        continue

                    # Parse shape data
                    shape_points = lane_child.attrib["shape"].split(" ")
                    node_ids = []

                    for point_str in shape_points:
                        try:
                            x, y = map(float, point_str.split(","))

                            # Convert coordinates
                            lon, lat = self._convert_coordinates(
                                conv_boundary, orig_boundary, x, y
                            )

                            # Convert to pixels
                            px_x, px_y = self.smopy_map.to_pixels(lat, lon)

                            # Calculate area index
                            index_x = max(
                                0,
                                min(
                                    int(abs(leftmost - lon) // x_division_size),
                                    self.settings.num_of_division - 1,
                                ),
                            )
                            index_y = max(
                                0,
                                min(
                                    int(abs(top - lat) // y_division_size),
                                    self.settings.num_of_division - 1,
                                ),
                            )

                            # Add node if not exists
                            if (px_x, px_y) not in node_mappings["xy_to_id"]:
                                graph.add_node(node_id, pos=(px_x, px_y))
                                node_mappings["xy_to_id"][(px_x, px_y)] = node_id
                                node_mappings["id_to_index"][node_id] = (
                                    index_x,
                                    index_y,
                                )
                                node_mappings["id_to_coordinate"][node_id] = {
                                    "longitude": lon,
                                    "latitude": lat,
                                }

                                # Update index_to_id mapping
                                if (index_x, index_y) not in node_mappings[
                                    "index_to_id"
                                ]:
                                    node_mappings["index_to_id"][
                                        (index_x, index_y)
                                    ] = []
                                node_mappings["index_to_id"][(index_x, index_y)].append(
                                    node_id
                                )

                                node_ids.append(node_id)
                                node_id += 1
                            else:
                                node_ids.append(node_mappings["xy_to_id"][(px_x, px_y)])

                        except (ValueError, KeyError):
                            continue

                    # Add edges
                    speed = float(lane_child.attrib.get("speed", 30.0))
                    for i in range(len(node_ids) - 1):
                        start_node = node_ids[i]
                        end_node = node_ids[i + 1]

                        # Calculate edge weight (distance)
                        start_pos = graph.nodes[start_node]["pos"]
                        end_pos = graph.nodes[end_node]["pos"]
                        weight = np.sqrt(
                            (end_pos[0] - start_pos[0]) ** 2
                            + (end_pos[1] - start_pos[1]) ** 2
                        )

                        graph.add_edge(start_node, end_node, weight=weight, speed=speed)

        # Extract largest strongly connected component
        scc = list(nx.strongly_connected_components(graph))
        if scc:
            largest_scc = max(scc, key=len)
            nodes_to_remove = set(graph.nodes()) - largest_scc
            graph.remove_nodes_from(nodes_to_remove)

            logger.info(f"Removed {len(nodes_to_remove)} nodes not in largest SCC")

        return graph, node_mappings

    def _convert_coordinates(
        self, conv_boundary: list[float], orig_boundary: list[float], x: float, y: float
    ) -> tuple[float, float]:
        """Convert SUMO coordinates to lat/lon."""
        orig_per_conv_x = abs(orig_boundary[0] - orig_boundary[2]) / abs(
            conv_boundary[0] - conv_boundary[2]
        )
        orig_per_conv_y = abs(orig_boundary[1] - orig_boundary[3]) / abs(
            conv_boundary[1] - conv_boundary[3]
        )

        lon = orig_boundary[0] + (x * orig_per_conv_x)
        lat = orig_boundary[1] + (y * orig_per_conv_y)

        return lon, lat

    def _is_non_roadway(self, edge_element) -> bool:
        """Check if edge represents non-roadway (railway, footway, etc.)."""
        attrs_str = str(edge_element.attrib)
        non_roadway_types = [
            "railway",
            "highway.cycleway",
            "highway.footway",
            "highway.living_street",
            "highway.path",
            "highway.pedestrian",
            "highway.step",
        ]

        return any(nrt in attrs_str for nrt in non_roadway_types)

    def _initialize_cars(self) -> None:
        """Initialize cars with random positions and destinations."""
        logger.info(f"Initializing {self.settings.number_of_cars} cars")

        nodes = list(self.road_graph.nodes())

        cars_created = 0
        attempts = 0
        max_attempts = self.settings.number_of_cars * 10

        while cars_created < self.settings.number_of_cars and attempts < max_attempts:
            attempts += 1

            try:
                # Select random origin and destination
                origin_node = random.choice(nodes)
                dest_node = random.choice(nodes)

                if origin_node == dest_node:
                    continue

                # Calculate shortest path
                try:
                    shortest_path = nx.dijkstra_path(
                        self.road_graph, origin_node, dest_node
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                if len(shortest_path) < 2:
                    continue

                # Create car
                car = Car(
                    orig_node_id=origin_node,
                    dest_node_id=dest_node,
                    shortest_path=shortest_path,
                    num_of_division=self.settings.num_of_division,
                )

                car.init(self.road_graph)
                self.cars_list.append(car)

                # Add to first edge
                first_edge = (shortest_path[0], shortest_path[1])
                if first_edge in self.edges_cars_dict:
                    self.edges_cars_dict[first_edge].append(car)

                cars_created += 1

            except Exception as e:
                logger.warning(f"Failed to create car {cars_created}: {e}")
                continue

        logger.info(
            f"Successfully created {cars_created} cars after {attempts} attempts"
        )

    def _setup_animation(self) -> None:
        """Setup matplotlib animation components."""
        logger.info("Setting up animation")

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(
            111, autoscale_on=False, xlim=self.map_bounds[:2], ylim=self.map_bounds[2:]
        )

        # Load and display map image
        map_image_path = self.settings.get_output_dir() / "map.png"
        if map_image_path.exists():
            img = Image.open(map_image_path)
            img_array = np.asarray(img)
            self.ax.imshow(img_array)

        # Create plot lines
        (self.line,) = plt.plot(
            [],
            [],
            color="green",
            marker="s",
            linestyle="",
            markersize=5,
            label="Empty Cars",
        )
        (self.ride_line,) = plt.plot(
            [],
            [],
            color="blue",
            marker="s",
            linestyle="",
            markersize=5,
            label="Occupied Cars",
        )
        (self.dest_line,) = plt.plot(
            [],
            [],
            color="red",
            marker="*",
            linestyle="",
            markersize=6,
            label="Empty Car Destinations",
        )
        (self.dest_ride_line,) = plt.plot(
            [],
            [],
            color="blue",
            marker="*",
            linestyle="",
            markersize=6,
            label="Occupied Car Destinations",
        )

        self.title_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top"
        )
        self.ax.invert_yaxis()
        self.ax.legend(loc="upper right")

        plt.tight_layout()

    def _run_simulation(self) -> None:
        """Run the main simulation loop."""
        logger.info("Starting simulation loop")

        if self.settings.save_animation and self.fig is not None:
            # Run with animation
            ani = FuncAnimation(
                self.fig,
                self._animate_step,
                frames=range(self.settings.max_steps),
                init_func=self._init_animation,
                blit=False,
                interval=self.settings.animation_interval,
            )

            # Save animation
            animation_file = (
                self.settings.get_output_dir()
                / f"simulation_{self.settings.epsilon}.mp4"
            )
            ani.save(str(animation_file), writer="ffmpeg", fps=20)
            logger.info(f"Saved animation to {animation_file}")
        else:
            # Run without animation
            for step in range(self.settings.max_steps):
                self._simulate_step(step)

                if step % 1000 == 0:
                    logger.info(f"Simulation step: {step}")

                if len(self.cars_list) == 0:
                    logger.info(f"All cars completed at step {step}")
                    break

    def _init_animation(self):
        """Initialize animation."""
        self.line.set_data([], [])
        self.ride_line.set_data([], [])
        self.dest_line.set_data([], [])
        self.dest_ride_line.set_data([], [])
        self.title_text.set_text("Simulation step: 0")
        return (
            self.line,
            self.ride_line,
            self.dest_line,
            self.dest_ride_line,
            self.title_text,
        )

    def _animate_step(self, frame):
        """Animate one simulation step."""
        self._simulate_step(frame)

        # Update visualization
        empty_cars = [
            (car.current_position[0], car.current_position[1])
            for car in self.cars_list
            if not car.ride_flag
        ]
        occupied_cars = [
            (car.current_position[0], car.current_position[1])
            for car in self.cars_list
            if car.ride_flag
        ]

        empty_dests = []
        occupied_dests = []

        for car in self.cars_list:
            if car.dest_node_id in self.node_mappings["id_to_coordinate"]:
                coord = self.node_mappings["id_to_coordinate"][car.dest_node_id]
                dest_x, dest_y = self.smopy_map.to_pixels(
                    coord["latitude"], coord["longitude"]
                )

                if car.ride_flag:
                    occupied_dests.append((dest_x, dest_y))
                else:
                    empty_dests.append((dest_x, dest_y))

        # Update plot data
        if empty_cars:
            x_data, y_data = zip(*empty_cars, strict=False)
            self.line.set_data(x_data, y_data)
        else:
            self.line.set_data([], [])

        if occupied_cars:
            x_data, y_data = zip(*occupied_cars, strict=False)
            self.ride_line.set_data(x_data, y_data)
        else:
            self.ride_line.set_data([], [])

        if empty_dests:
            x_data, y_data = zip(*empty_dests, strict=False)
            self.dest_line.set_data(x_data, y_data)
        else:
            self.dest_line.set_data([], [])

        if occupied_dests:
            x_data, y_data = zip(*occupied_dests, strict=False)
            self.dest_ride_line.set_data(x_data, y_data)
        else:
            self.dest_ride_line.set_data([], [])

        self.title_text.set_text(
            f"Step: {frame}, Cars: {len(self.cars_list)}, Time: {self.current_time_slot:02d}:00"
        )

        return (
            self.line,
            self.ride_line,
            self.dest_line,
            self.dest_ride_line,
            self.title_text,
        )

    def _simulate_step(self, step: int) -> None:
        """Simulate one time step."""
        self.animation_count += 1

        # Update time slot (2880 steps = 1 hour)
        if self.animation_count % 2880 == 0:
            self.current_time_slot = (self.current_time_slot + 1) % 24

        # Move all cars and handle arrivals
        cars_to_remove = []
        cars_to_add = []
        reward_sum = 0.0

        for car in self.cars_list:
            reward_sum += car.total_reward

            # Move car
            x_new, y_new, arrived = car.move(
                self.road_graph, self.edges_cars_dict, self.settings.sensitivity
            )

            # Handle arrival
            if arrived:
                new_car = self._handle_car_arrival(car, step)
                if new_car:
                    cars_to_remove.append(car)
                    cars_to_add.append(new_car)

        # Update car list
        for car in cars_to_remove:
            if car in self.cars_list:
                self.cars_list.remove(car)

        for car in cars_to_add:
            self.cars_list.append(car)

        # Record metrics
        if self.cars_list:
            avg_reward = reward_sum / len(self.cars_list)
            self.total_rewards.append(avg_reward)

        # Cleanup
        if step % 1000 == 0:
            gc.collect()

    def _handle_car_arrival(self, car: Car, step: int) -> Car:
        """Handle car arrival at destination and create new car."""
        try:
            # Get current area
            current_area = self.node_mappings["id_to_index"][car.dest_node_id]
            index_x, index_y = current_area

            # Update experience
            car.update_experience(current_area, self.current_time_slot, 0.0)

            # Determine next destination
            ride_flag, dest_node_id, reward = self._choose_next_destination(
                car, index_x, index_y, self.current_time_slot
            )

            # Calculate shortest path
            try:
                shortest_path = nx.dijkstra_path(
                    self.road_graph, car.dest_node_id, dest_node_id
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Fallback to random destination
                dest_node_id = self._choose_random_destination(car.dest_node_id)
                shortest_path = nx.dijkstra_path(
                    self.road_graph, car.dest_node_id, dest_node_id
                )
                ride_flag = False
                reward = 0.0

            if len(shortest_path) < 2:
                dest_node_id = self._choose_random_destination(car.dest_node_id)
                shortest_path = nx.dijkstra_path(
                    self.road_graph, car.dest_node_id, dest_node_id
                )
                ride_flag = False
                reward = 0.0

            # Calculate movement cost
            orig_coord = self.node_mappings["id_to_coordinate"][car.dest_node_id]
            dest_coord = self.node_mappings["id_to_coordinate"][dest_node_id]

            movement_cost = self.reward_calculator.calculate_movement_cost(
                dist_on_sphere(
                    orig_coord["latitude"],
                    orig_coord["longitude"],
                    dest_coord["latitude"],
                    dest_coord["longitude"],
                )
            )

            # Update experience with reward and cost
            net_reward = reward - movement_cost
            car.update_experience(current_area, self.current_time_slot, net_reward)

            # Record trajectory
            self.car_trajectories.append(
                (
                    id(car),
                    step,
                    (orig_coord["latitude"], orig_coord["longitude"]),
                    (dest_coord["latitude"], dest_coord["longitude"]),
                )
            )

            # Create new car
            new_car = Car(
                orig_node_id=car.dest_node_id,
                dest_node_id=dest_node_id,
                shortest_path=shortest_path,
                num_of_division=self.settings.num_of_division,
            )

            # Transfer experience and state
            new_car.experience = car.experience
            new_car.total_reward = car.total_reward
            new_car.ride_flag = ride_flag

            # Initialize new car
            new_car.init(self.road_graph)

            # Add to edge tracking
            first_edge = (shortest_path[0], shortest_path[1])
            if first_edge in self.edges_cars_dict:
                self.edges_cars_dict[first_edge].append(new_car)

            return new_car

        except Exception as e:
            logger.error(f"Error handling car arrival: {e}")
            return None

    def _choose_next_destination(
        self, car: Car, current_x: int, current_y: int, time_slot: int
    ) -> tuple[bool, int, float]:
        """Choose next destination using epsilon-greedy strategy."""
        # Check for passenger pickup
        if (
            current_y < len(self.ride_probabilities[time_slot])
            and current_x < len(self.ride_probabilities[time_slot][current_y])
            and self.ride_probabilities[time_slot][current_y][current_x]
            >= random.random()
            and current_y < len(self.reward_areas)
            and current_x < len(self.reward_areas[current_y])
            and len(self.reward_areas[current_y][current_x]) > 0
        ):

            # Pick up passenger
            passenger_data = random.choice(self.reward_areas[current_y][current_x])
            dest_x, dest_y = passenger_data["index_x"], passenger_data["index_y"]
            reward = passenger_data["reward"]

            # Get destination node
            if (dest_x, dest_y) in self.node_mappings["index_to_id"]:
                dest_nodes = self.node_mappings["index_to_id"][(dest_x, dest_y)]
                dest_node_id = random.choice(dest_nodes)
                return True, dest_node_id, reward

        # Epsilon-greedy decision
        if random.random() < self.settings.epsilon:
            # Exploration: random destination
            dest_node_id = self._choose_random_destination(car.orig_node_id)
            return False, dest_node_id, 0.0
        else:
            # Exploitation: use experience
            best_area = car.get_best_area(time_slot)
            if best_area and best_area in self.node_mappings["index_to_id"]:
                dest_nodes = self.node_mappings["index_to_id"][best_area]
                dest_node_id = random.choice(dest_nodes)
                return False, dest_node_id, 0.0
            else:
                # No experience, choose randomly
                dest_node_id = self._choose_random_destination(car.orig_node_id)
                return False, dest_node_id, 0.0

    def _choose_random_destination(self, avoid_node_id: int) -> int:
        """Choose a random destination node."""
        nodes = list(self.road_graph.nodes())
        available_nodes = [n for n in nodes if n != avoid_node_id]
        return random.choice(available_nodes) if available_nodes else avoid_node_id

    def _save_results(self) -> None:
        """Save all simulation results."""
        logger.info("Saving simulation results")

        simulation_params = {
            "number_of_cars": self.settings.number_of_cars,
            "epsilon": self.settings.epsilon,
            "max_steps": self.settings.max_steps,
            "num_divisions": self.settings.num_of_division,
            "sensitivity": self.settings.sensitivity,
            "network_file": self.settings.network_file,
            "geojson_file": self.settings.geojson_file,
            "taxi_data_dir": self.settings.taxi_data_dir,
        }

        self.output_manager.save_simulation_results(
            self.cars_list,
            self.total_rewards,
            self.car_trajectories,
            simulation_params,
            self.settings.num_of_division,
        )

        self.output_manager.create_final_report(
            self.cars_list, self.total_rewards, simulation_params
        )
