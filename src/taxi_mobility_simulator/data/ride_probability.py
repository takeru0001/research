"""Ride probability calculation from taxi data."""

import datetime as dt
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from .reward_calculator import RewardCalculator

logger = logging.getLogger(__name__)


class RideProbabilityCalculator:
    """Calculate ride probabilities from historical taxi data."""

    def __init__(self, reward_calculator: RewardCalculator):
        """Initialize with a reward calculator."""
        self.reward_calculator = reward_calculator
        self.pst_timezone = dt.timezone(dt.timedelta(hours=-8), "PST")

    def calculate_ride_probabilities(
        self, network_file: Path, taxi_data_dir: Path, num_divisions: int
    ) -> tuple[list[list[list[float]]], list[list[list[dict]]]]:
        """
        Calculate ride probabilities and rewards for each area and time slot.

        Returns:
            Tuple of (ride_probabilities, reward_data)
            - ride_probabilities: [time][y][x] -> probability (0.0-1.0)
            - reward_data: [y][x] -> list of reward dictionaries
        """
        logger.info("Starting ride probability calculation")

        # Parse network file for boundaries
        boundaries = self._parse_network_boundaries(network_file)

        # Get taxi data files
        taxi_files = self._get_taxi_files(taxi_data_dir)

        # Extract ride points and rewards
        ride_points, reward_list = self._extract_ride_data(taxi_files)

        # Calculate area-based statistics
        ride_counts, reward_areas = self._calculate_area_statistics(
            boundaries, ride_points, reward_list, num_divisions
        )

        # Convert counts to probabilities
        ride_probabilities = self._calculate_probabilities(ride_counts, num_divisions)

        logger.info(f"Calculated probabilities for {len(ride_points)} ride points")
        return ride_probabilities, reward_areas

    def _parse_network_boundaries(
        self, network_file: Path
    ) -> tuple[list[float], list[float]]:
        """Parse SUMO network file to extract boundaries."""
        try:
            tree = ET.parse(network_file)
            root = tree.getroot()

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

        except (ET.ParseError, FileNotFoundError, KeyError) as e:
            logger.error(f"Error parsing network file {network_file}: {e}")
            raise

    def _get_taxi_files(self, taxi_data_dir: Path) -> list[Path]:
        """Get list of taxi data files."""
        taxi_list_file = taxi_data_dir / "_cabs.txt"

        if not taxi_list_file.exists():
            raise FileNotFoundError(f"Taxi list file not found: {taxi_list_file}")

        taxi_files = []
        try:
            with taxi_list_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Extract filename from quoted string
                        filename = line.split('"')[1] if '"' in line else line
                        taxi_file = taxi_data_dir / f"new_{filename}.txt"
                        if taxi_file.exists():
                            taxi_files.append(taxi_file)
                        else:
                            logger.warning(f"Taxi file not found: {taxi_file}")

            logger.info(f"Found {len(taxi_files)} taxi data files")
            return taxi_files

        except (OSError, IndexError) as e:
            logger.error(f"Error reading taxi list file: {e}")
            raise

    def _extract_ride_data(
        self, taxi_files: list[Path]
    ) -> tuple[
        list[dict], list[tuple[float, tuple[float, float], tuple[float, float], int]]
    ]:
        """Extract ride points and reward data from taxi files."""
        ride_points = []
        reward_list = []

        for taxi_file in taxi_files:
            try:
                file_ride_points, file_rewards = self._process_taxi_file(taxi_file)
                ride_points.extend(file_ride_points)
                reward_list.extend(file_rewards)

            except Exception as e:
                logger.warning(f"Error processing taxi file {taxi_file}: {e}")
                continue

        logger.info(
            f"Extracted {len(ride_points)} ride points and {len(reward_list)} rewards"
        )
        return ride_points, reward_list

    def _process_taxi_file(
        self, taxi_file: Path
    ) -> tuple[
        list[dict], list[tuple[float, tuple[float, float], tuple[float, float], int]]
    ]:
        """Process a single taxi data file."""
        ride_points = []
        reward_list = []

        # Read and parse data
        data_points = []
        with taxi_file.open() as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        data_point = {
                            "latitude": float(parts[0]),
                            "longitude": float(parts[1]),
                            "ride_state": int(parts[2]),
                            "unixtime": int(parts[3]),
                        }
                        data_points.append(data_point)
                except (ValueError, IndexError):
                    continue

        # Sort by timestamp
        data_points.sort(key=lambda x: x["unixtime"])

        # Extract rides
        prev_state = None
        current_ride = []

        for point in data_points:
            current_state = point["ride_state"]

            if prev_state is None:
                prev_state = current_state
                if current_state == 1:  # Start of ride
                    ride_points.append(point)
                    current_ride = [point]
                continue

            # State transition logic
            if prev_state == 0 and current_state == 1:
                # Start of new ride
                ride_points.append(point)
                current_ride = [point]

            elif prev_state == 1 and current_state == 0:
                # End of ride
                current_ride.append(point)

                # Calculate reward for this ride
                if len(current_ride) >= 2:
                    reward_data = self.reward_calculator.calculate_ride_reward(
                        current_ride
                    )
                    if reward_data[0] is not None:
                        reward, origin, destination = reward_data
                        duration = (
                            current_ride[-1]["unixtime"] - current_ride[0]["unixtime"]
                        )

                        # Filter valid rides (2min < duration < 40min, reward < $40)
                        if 120 < duration < 2400 and reward < 40:
                            reward_list.append((reward, origin, destination, duration))
                        else:
                            # Remove invalid ride point
                            if ride_points and ride_points[-1] == current_ride[0]:
                                ride_points.pop()

                current_ride = []

            elif current_state == 1:
                # Continue ride
                current_ride.append(point)

            prev_state = current_state

        return ride_points, reward_list

    def _calculate_area_statistics(
        self,
        boundaries: tuple[list[float], list[float]],
        ride_points: list[dict],
        reward_list: list[tuple],
        num_divisions: int,
    ) -> tuple[list[list[list[int]]], list[list[list[dict]]]]:
        """Calculate ride counts and rewards for each area."""
        conv_boundary, orig_boundary = boundaries

        # Calculate area dimensions
        top, bottom = orig_boundary[3], orig_boundary[1]
        leftmost, rightmost = orig_boundary[0], orig_boundary[2]
        x_division_size = abs(leftmost - rightmost) / num_divisions
        y_division_size = abs(top - bottom) / num_divisions

        # Initialize data structures
        ride_counts = [
            [[0 for _ in range(num_divisions)] for _ in range(num_divisions)]
            for _ in range(24)
        ]
        reward_areas = [
            [[] for _ in range(num_divisions)] for _ in range(num_divisions)
        ]

        # Process ride points
        for point in ride_points:
            try:
                longitude, latitude = point["longitude"], point["latitude"]
                unixtime = point["unixtime"]

                # Calculate area indices
                index_x = int(abs(leftmost - longitude) // x_division_size)
                index_y = int(abs(top - latitude) // y_division_size)

                # Clamp to valid range
                index_x = max(0, min(index_x, num_divisions - 1))
                index_y = max(0, min(index_y, num_divisions - 1))

                # Get time slot
                time_slot = dt.datetime.fromtimestamp(unixtime, self.pst_timezone).hour

                # Increment ride count
                ride_counts[time_slot][index_y][index_x] += 1

            except (KeyError, ValueError, OSError):
                continue

        # Process reward data
        for reward, origin, destination, duration in reward_list:
            try:
                orig_lat, orig_lon = origin
                dest_lat, dest_lon = destination

                # Calculate origin area
                orig_x = int(abs(leftmost - orig_lon) // x_division_size)
                orig_y = int(abs(top - orig_lat) // y_division_size)

                # Calculate destination area
                dest_x = int(abs(leftmost - dest_lon) // x_division_size)
                dest_y = int(abs(top - dest_lat) // y_division_size)

                # Clamp to valid range
                orig_x = max(0, min(orig_x, num_divisions - 1))
                orig_y = max(0, min(orig_y, num_divisions - 1))
                dest_x = max(0, min(dest_x, num_divisions - 1))
                dest_y = max(0, min(dest_y, num_divisions - 1))

                # Add reward data
                reward_data = {
                    "reward": reward,
                    "index_x": dest_x,
                    "index_y": dest_y,
                    "elapsed_time": duration,
                }
                reward_areas[orig_y][orig_x].append(reward_data)

            except (ValueError, IndexError):
                continue

        return ride_counts, reward_areas

    def _calculate_probabilities(
        self, ride_counts: list[list[list[int]]], num_divisions: int
    ) -> list[list[list[float]]]:
        """Convert ride counts to probabilities."""
        probabilities = [
            [[0.0 for _ in range(num_divisions)] for _ in range(num_divisions)]
            for _ in range(24)
        ]

        for time_slot in range(24):
            # Find maximum count for this time slot
            max_count = 0
            for y in range(num_divisions):
                for x in range(num_divisions):
                    max_count = max(max_count, ride_counts[time_slot][y][x])

            # Convert to probabilities
            if max_count > 0:
                prob_per_ride = 1.0 / max_count
                for y in range(num_divisions):
                    for x in range(num_divisions):
                        probabilities[time_slot][y][x] = (
                            ride_counts[time_slot][y][x] * prob_per_ride
                        )

        return probabilities
