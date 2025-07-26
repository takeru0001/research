"""Configuration settings for the taxi mobility simulator."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SimulationSettings(BaseSettings):
    """Simulation configuration settings."""

    # File paths
    network_file: Path = Field(
        default=Path("data/maps/EntireSanFrancisco.net.xml"),
        description="Path to the SUMO network XML file",
    )
    geojson_file: Path = Field(
        default=Path("data/maps/EntireSanFrancisco.geojson"),
        description="Path to the GeoJSON boundary file",
    )
    taxi_data_dir: Path = Field(
        default=Path("data/raw/cabspottingdata"),
        description="Directory containing taxi data files",
    )
    taxi_list_file: Path = Field(
        default=Path("data/raw/cabspottingdata/_cabs.txt"),
        description="File listing available taxi data files",
    )

    # Simulation parameters
    number_of_cars: int = Field(
        default=100, ge=1, le=10000, description="Number of cars in the simulation"
    )
    num_of_division: int = Field(
        default=10, ge=2, le=100, description="Number of divisions for area grid"
    )
    epsilon: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Epsilon value for epsilon-greedy strategy",
    )
    max_steps: int = Field(
        default=150000, ge=1, description="Maximum number of simulation steps"
    )
    sensitivity: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Movement sensitivity parameter"
    )

    # Animation settings
    animation_interval: int = Field(
        default=50, ge=1, description="Animation interval in milliseconds"
    )
    save_animation: bool = Field(
        default=True, description="Whether to save animation as video"
    )

    # Output settings
    output_dir: Path | None = Field(
        default=None, description="Output directory (auto-generated if None)"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    # Map settings
    map_zoom_level: int = Field(
        default=17, ge=1, le=20, description="Map zoom level for tile server"
    )
    tile_server: str = Field(
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        description="Tile server URL",
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_prefix = "TAXI_SIM_"
        case_sensitive = False

    def get_output_dir(self) -> Path:
        """Get or create output directory."""
        if self.output_dir is None:
            from datetime import date

            self.output_dir = Path(f"output/simulation_{self.epsilon}_{date.today()}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
