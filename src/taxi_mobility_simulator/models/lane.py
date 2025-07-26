"""Lane model for road network representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .car import Car


@dataclass
class Lane:
    """Represents a lane in the road network."""

    from_id: str | None = None
    to_id: str | None = None
    speed: float = 0.0
    node_id_list: list[int] = field(default_factory=list)
    node_x_list: list[float] = field(default_factory=list)
    node_y_list: list[float] = field(default_factory=list)
    car_list: list[Car] = field(default_factory=list)

    def add_from_to(self, from_id: str, to_id: str) -> None:
        """Set the from and to node IDs."""
        self.from_id = from_id
        self.to_id = to_id

    def set_geometry(
        self,
        speed: float,
        node_id_list: list[int],
        node_x_list: list[float],
        node_y_list: list[float],
    ) -> None:
        """Set the geometric properties of the lane."""
        if len(node_id_list) != len(node_x_list) or len(node_x_list) != len(
            node_y_list
        ):
            raise ValueError("All node lists must have the same length")

        self.speed = speed
        self.node_id_list = node_id_list.copy()
        self.node_x_list = node_x_list.copy()
        self.node_y_list = node_y_list.copy()

    def get_length(self) -> float:
        """Calculate the total length of the lane."""
        if len(self.node_x_list) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(self.node_x_list) - 1):
            dx = self.node_x_list[i + 1] - self.node_x_list[i]
            dy = self.node_y_list[i + 1] - self.node_y_list[i]
            total_length += (dx**2 + dy**2) ** 0.5

        return total_length

    def is_valid(self) -> bool:
        """Check if the lane has valid geometry."""
        return (
            len(self.node_id_list) >= 2
            and len(self.node_x_list) >= 2
            and len(self.node_y_list) >= 2
            and self.speed > 0
        )
