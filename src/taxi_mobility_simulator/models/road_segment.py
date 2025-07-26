"""Road segment model for U-turn handling."""

from dataclasses import dataclass

from .lane import Lane


@dataclass
class RoadSegment:
    """Represents a bidirectional road segment for U-turn operations."""

    lane1: Lane
    lane2: Lane

    def __post_init__(self) -> None:
        """Validate that the lanes form a valid bidirectional segment."""
        if not self._is_valid_segment():
            raise ValueError("Lanes do not form a valid bidirectional segment")

    def _is_valid_segment(self) -> bool:
        """Check if the two lanes form a valid bidirectional segment."""
        return (
            self.lane1.from_id == self.lane2.to_id
            and self.lane1.to_id == self.lane2.from_id
            and self.lane1.is_valid()
            and self.lane2.is_valid()
        )

    def get_reverse_lane(self, current_lane: Lane) -> Lane | None:
        """Get the reverse lane for U-turn operations."""
        if current_lane == self.lane1:
            return self.lane2
        elif current_lane == self.lane2:
            return self.lane1
        else:
            return None

    def allows_uturn(self) -> bool:
        """Check if U-turn is allowed on this road segment."""
        # This could be extended with more sophisticated rules
        return True

    def get_average_speed(self) -> float:
        """Get the average speed limit of both lanes."""
        return (self.lane1.speed + self.lane2.speed) / 2.0
