"""Reward calculation utilities."""

import logging
from math import acos, cos, radians, sin

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6378.137


def latlng_to_xyz(lat: float, lng: float) -> tuple[float, float, float]:
    """Convert latitude/longitude to XYZ coordinates."""
    rlat, rlng = radians(lat), radians(lng)
    coslat = cos(rlat)
    return coslat * cos(rlng), coslat * sin(rlng), sin(rlat)


def dist_on_sphere(
    pos0_latitude: float,
    pos0_longitude: float,
    pos1_latitude: float,
    pos1_longitude: float,
    radius: float = EARTH_RADIUS_KM,
) -> float:
    """Calculate distance between two points on a sphere."""
    pos0 = (pos0_latitude, pos0_longitude)
    pos1 = (pos1_latitude, pos1_longitude)

    if pos0 == pos1:
        return 0.0

    try:
        xyz0, xyz1 = latlng_to_xyz(*pos0), latlng_to_xyz(*pos1)
        dot_product = sum(x * y for x, y in zip(xyz0, xyz1, strict=False))

        # Clamp to avoid numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))

        return acos(dot_product) * radius
    except (ValueError, OverflowError) as e:
        logger.error(f"Error calculating distance: {e}")
        return 0.0


def calculate_speed(unixtime_difference: int, moving_distance: float) -> float:
    """Calculate speed from time difference and distance."""
    if unixtime_difference <= 0:
        return 0.0

    elapsed_hours = unixtime_difference / 3600.0
    return moving_distance / elapsed_hours if elapsed_hours > 0 else 0.0


class RewardCalculator:
    """Calculate rewards for taxi rides."""

    def __init__(
        self,
        base_fare: float = 2.5,
        per_km_rate: float = 1.8,
        per_minute_rate: float = 0.4,
        fuel_cost_per_km: float = 0.1,
        max_speed_kmh: float = 140.0,
    ):
        """Initialize reward calculator with pricing parameters."""
        self.base_fare = base_fare
        self.per_km_rate = per_km_rate
        self.per_minute_rate = per_minute_rate
        self.fuel_cost_per_km = fuel_cost_per_km
        self.max_speed_kmh = max_speed_kmh

    def calculate_ride_reward(
        self, coordinates_in_ride: list[dict[str, float]]
    ) -> tuple[float | None, tuple[float, float] | None, tuple[float, float] | None]:
        """
        Calculate reward for a taxi ride.

        Returns:
            Tuple of (reward, origin_coords, destination_coords) or (None, None, None) if invalid
        """
        if len(coordinates_in_ride) < 2:
            logger.warning("Insufficient coordinates for ride calculation")
            return None, None, None

        try:
            # Get origin and destination
            origin = coordinates_in_ride[0]
            destination = coordinates_in_ride[-1]

            origin_coords = (origin["latitude"], origin["longitude"])
            dest_coords = (destination["latitude"], destination["longitude"])

            # Calculate total distance
            total_distance = 0.0
            for i in range(len(coordinates_in_ride) - 1):
                current = coordinates_in_ride[i]
                next_point = coordinates_in_ride[i + 1]

                segment_distance = dist_on_sphere(
                    current["latitude"],
                    current["longitude"],
                    next_point["latitude"],
                    next_point["longitude"],
                )
                total_distance += segment_distance

            # Calculate time duration
            start_time = coordinates_in_ride[0]["unixtime"]
            end_time = coordinates_in_ride[-1]["unixtime"]
            duration_seconds = end_time - start_time
            duration_minutes = duration_seconds / 60.0

            # Validate ride (check for unrealistic speeds)
            if duration_seconds > 0:
                avg_speed = calculate_speed(duration_seconds, total_distance)
                if avg_speed > self.max_speed_kmh:
                    # logger.warning(f"Unrealistic speed detected: {avg_speed:.2f} km/h")
                    return None, None, None

            # Calculate fare
            fare = self._calculate_fare(total_distance, duration_minutes)

            # Calculate costs
            fuel_cost = total_distance * self.fuel_cost_per_km

            # Net reward
            reward = fare - fuel_cost

            logger.debug(
                f"Ride reward calculated: distance={total_distance:.2f}km, "
                f"duration={duration_minutes:.1f}min, fare=${fare:.2f}, "
                f"fuel_cost=${fuel_cost:.2f}, reward=${reward:.2f}"
            )

            return reward, origin_coords, dest_coords

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error calculating ride reward: {e}")
            return None, None, None

    def _calculate_fare(self, distance_km: float, duration_minutes: float) -> float:
        """Calculate taxi fare based on distance and time."""
        distance_fare = distance_km * self.per_km_rate
        time_fare = duration_minutes * self.per_minute_rate

        return self.base_fare + distance_fare + time_fare

    def calculate_movement_cost(self, distance_km: float) -> float:
        """Calculate cost of movement (fuel, wear, etc.)."""
        return distance_km * self.fuel_cost_per_km
