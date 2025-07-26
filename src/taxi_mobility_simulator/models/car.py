"""Car model for taxi mobility simulation."""

import logging
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

logger = logging.getLogger(__name__)


class RideState(Enum):
    """Ride state enumeration."""

    EMPTY = 0
    OCCUPIED = 1


@dataclass
class ExperienceData:
    """Experience data for a specific area and time."""

    reward: float = 0.0
    count: int = 0
    step: int = 0
    reward_per_step: float = 0.0


@dataclass
class Car:
    """Car model with reinforcement learning capabilities."""

    orig_node_id: int
    dest_node_id: int
    shortest_path: list[int]
    num_of_division: int

    # Experience data: [time_slot][area_coordinates] -> ExperienceData
    experience: list[dict[tuple[int, int], ExperienceData]] = field(
        default_factory=lambda: [{} for _ in range(24)]
    )

    # Current state
    current_sp_index: int = 0
    current_speed: float = 0.0
    current_start_node: list[float] = field(default_factory=list)
    current_position: list[float] = field(default_factory=list)
    current_end_node: list[float] = field(default_factory=list)
    current_distance: float = 0.0
    goal_arrived: bool = False
    ride_flag: bool = False

    # Learning metrics
    num_of_elapsed_steps: int = 0
    max_reward_per_step_index: tuple[int, int] | None = None
    total_reward: float = 0.0

    def init(self, graph: nx.DiGraph) -> None:
        """Initialize car position and state."""
        try:
            if len(self.shortest_path) < 2:
                raise ValueError("Shortest path must have at least 2 nodes")

            start_node = self.shortest_path[0]
            end_node = self.shortest_path[1]

            if not graph.has_edge(start_node, end_node):
                raise ValueError(f"Edge {start_node}->{end_node} not found in graph")

            # Get node positions
            start_pos = graph.nodes[start_node]["pos"]
            end_pos = graph.nodes[end_node]["pos"]

            self.current_start_node = list(start_pos)
            self.current_end_node = list(end_pos)
            self.current_position = list(start_pos)
            self.current_distance = graph.edges[start_node, end_node]["weight"]

            logger.debug(f"Car initialized at node {start_node}, heading to {end_node}")

        except (KeyError, ValueError) as e:
            logger.error(f"Failed to initialize car: {e}")
            raise

    def move(
        self,
        graph: nx.DiGraph,
        edges_cars_dict: dict[tuple[int, int], list["Car"]],
        sensitivity: float,
    ) -> tuple[float, float, bool]:
        """Move car one step along its path."""
        try:
            # Calculate optimal velocity (simplified)
            target_speed = self._calculate_optimal_velocity(
                graph, edges_cars_dict, sensitivity
            )

            # Update position
            self.current_speed = target_speed

            # Move along current edge
            if self.current_distance > 0:
                move_distance = min(self.current_speed, self.current_distance)
                progress = move_distance / self.current_distance

                # Linear interpolation between start and end nodes
                new_x = self.current_start_node[0] + progress * (
                    self.current_end_node[0] - self.current_start_node[0]
                )
                new_y = self.current_start_node[1] + progress * (
                    self.current_end_node[1] - self.current_start_node[1]
                )

                self.current_position = [new_x, new_y]
                self.current_distance -= move_distance

                # Check if reached end of current edge
                if self.current_distance <= 0.1:  # Small threshold for floating point
                    self._advance_to_next_edge(graph, edges_cars_dict)

            self.num_of_elapsed_steps += 1
            return self.current_position[0], self.current_position[1], self.goal_arrived

        except Exception as e:
            logger.error(f"Error moving car: {e}")
            return self.current_position[0], self.current_position[1], False

    def _calculate_optimal_velocity(
        self,
        graph: nx.DiGraph,
        edges_cars_dict: dict[tuple[int, int], list["Car"]],
        sensitivity: float,
    ) -> float:
        """Calculate optimal velocity based on traffic and road conditions."""
        try:
            current_edge = (
                self.shortest_path[self.current_sp_index],
                self.shortest_path[self.current_sp_index + 1],
            )

            # Get road speed limit
            max_speed = graph.edges[current_edge]["speed"]

            # Count cars on current edge
            cars_on_edge = len(edges_cars_dict.get(current_edge, []))

            # Simple traffic model: reduce speed based on car density
            traffic_factor = max(0.1, 1.0 - (cars_on_edge * 0.1))

            # Apply sensitivity
            optimal_speed = max_speed * traffic_factor * sensitivity

            return max(0.1, optimal_speed)  # Minimum speed

        except (KeyError, IndexError) as e:
            logger.warning(f"Error calculating optimal velocity: {e}")
            return 1.0  # Default speed

    def _advance_to_next_edge(
        self, graph: nx.DiGraph, edges_cars_dict: dict[tuple[int, int], list["Car"]]
    ) -> None:
        """Advance to the next edge in the shortest path."""
        try:
            # Remove from current edge
            current_edge = (
                self.shortest_path[self.current_sp_index],
                self.shortest_path[self.current_sp_index + 1],
            )
            if self in edges_cars_dict.get(current_edge, []):
                edges_cars_dict[current_edge].remove(self)

            # Move to next edge
            self.current_sp_index += 1

            # Check if reached destination
            if self.current_sp_index >= len(self.shortest_path) - 1:
                self.goal_arrived = True
                logger.debug(f"Car reached destination: {self.dest_node_id}")
                return

            # Set up next edge
            next_start = self.shortest_path[self.current_sp_index]
            next_end = self.shortest_path[self.current_sp_index + 1]
            next_edge = (next_start, next_end)

            if not graph.has_edge(next_start, next_end):
                logger.error(f"Next edge {next_edge} not found in graph")
                self.goal_arrived = True
                return

            # Update position and distance
            self.current_start_node = list(graph.nodes[next_start]["pos"])
            self.current_end_node = list(graph.nodes[next_end]["pos"])
            self.current_position = list(self.current_start_node)
            self.current_distance = graph.edges[next_edge]["weight"]

            # Add to new edge
            edges_cars_dict[next_edge].append(self)

        except (KeyError, IndexError) as e:
            logger.error(f"Error advancing to next edge: {e}")
            self.goal_arrived = True

    def update_experience(
        self, area_coords: tuple[int, int], time_slot: int, reward: float
    ) -> None:
        """Update experience data for a specific area and time."""
        if time_slot < 0 or time_slot >= 24:
            logger.warning(f"Invalid time slot: {time_slot}")
            return

        if area_coords not in self.experience[time_slot]:
            self.experience[time_slot][area_coords] = ExperienceData()

        exp_data = self.experience[time_slot][area_coords]
        exp_data.reward += reward
        exp_data.step += self.num_of_elapsed_steps
        exp_data.count += 1

        # Update reward per step
        if exp_data.step > 0:
            exp_data.reward_per_step = exp_data.reward / exp_data.step

        self.total_reward += reward

        logger.debug(
            f"Updated experience for area {area_coords}, time {time_slot}: "
            f"reward={reward}, total_reward={self.total_reward}"
        )

    def get_best_area(self, time_slot: int) -> tuple[int, int] | None:
        """Get the area with the best reward per step for a given time slot."""
        if time_slot < 0 or time_slot >= 24:
            return None

        if not self.experience[time_slot]:
            return None

        best_area = max(
            self.experience[time_slot].items(), key=lambda x: x[1].reward_per_step
        )

        return best_area[0]
