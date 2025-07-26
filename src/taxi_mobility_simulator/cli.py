"""Command-line interface for the taxi mobility simulator."""

import logging
from pathlib import Path

import click

from .config.settings import SimulationSettings
from .simulation.simulator import TaxiSimulator


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_dir / "simulation.log")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


@click.command()
@click.option(
    "--cars",
    default=100,
    help="Number of cars in the simulation",
    type=click.IntRange(1, 10000),
)
@click.option(
    "--epsilon",
    default=0.1,
    help="Epsilon value for epsilon-greedy strategy",
    type=click.FloatRange(0.0, 1.0),
)
@click.option(
    "--steps",
    default=150000,
    help="Maximum number of simulation steps",
    type=click.IntRange(1),
)
@click.option(
    "--divisions",
    default=10,
    help="Number of area divisions",
    type=click.IntRange(2, 100),
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), help="Output directory for results"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option("--no-animation", is_flag=True, help="Disable animation saving")
def run_simulation(
    cars: int,
    epsilon: float,
    steps: int,
    divisions: int,
    config: Path | None,
    output_dir: Path | None,
    log_level: str,
    no_animation: bool,
) -> None:
    """Run the taxi mobility simulation."""
    # Set up logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        if config:
            settings = SimulationSettings(_env_file=config)
        else:
            settings = SimulationSettings()

        # Override with command line arguments
        settings.number_of_cars = cars
        settings.epsilon = epsilon
        settings.max_steps = steps
        settings.num_of_division = divisions
        settings.log_level = log_level
        settings.save_animation = not no_animation

        if output_dir:
            settings.output_dir = output_dir

        logger.info(f"Starting simulation with {cars} cars, epsilon={epsilon}")
        logger.info(f"Max steps: {steps}, Area divisions: {divisions}")

        # Create and run simulator
        simulator = TaxiSimulator(settings)
        simulator.run()

        logger.info("Simulation completed successfully")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise click.ClickException(f"Simulation failed: {e}")


@click.group()
def main() -> None:
    """Taxi Mobility Simulator - Reinforcement Learning Based Movement Model."""
    pass


@main.command()
@click.option(
    "--network-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="SUMO network XML file",
)
@click.option(
    "--taxi-data-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing taxi data",
)
@click.option(
    "--divisions",
    default=10,
    help="Number of area divisions",
    type=click.IntRange(2, 100),
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for probability maps",
)
def analyze_data(
    network_file: Path, taxi_data_dir: Path, divisions: int, output_dir: Path | None
) -> None:
    """Analyze taxi data and generate probability maps."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        from .data.reward_calculator import RewardCalculator
        from .data.ride_probability import RideProbabilityCalculator
        from .utils.visualization import Visualizer

        # Create output directory
        if output_dir is None:
            output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting data analysis")

        # Calculate probabilities
        reward_calc = RewardCalculator()
        prob_calc = RideProbabilityCalculator(reward_calc)

        ride_probs, reward_areas = prob_calc.calculate_ride_probabilities(
            network_file, taxi_data_dir, divisions
        )

        # Generate visualizations
        visualizer = Visualizer()
        visualizer.create_probability_heatmaps(ride_probs, output_dir)

        logger.info(f"Analysis completed. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        raise click.ClickException(f"Data analysis failed: {e}")


# Add the run_simulation command to the main group
main.add_command(run_simulation, name="run")


if __name__ == "__main__":
    main()
