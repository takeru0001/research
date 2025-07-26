"""Output management utilities."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .visualization import Visualizer

logger = logging.getLogger(__name__)


class OutputManager:
    """Manage simulation output and result saving."""

    def __init__(self, output_dir: Path, visualizer: Visualizer):
        """Initialize output manager."""
        self.output_dir = output_dir
        self.visualizer = visualizer
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_simulation_results(
        self,
        cars_list: list,
        total_rewards: list[float],
        car_trajectories: list[tuple],
        simulation_params: dict[str, Any],
        num_divisions: int,
    ) -> None:
        """Save all simulation results."""
        logger.info(f"Saving simulation results to {self.output_dir}")

        try:
            # Save reward timeline
            self._save_reward_timeline(total_rewards, simulation_params["epsilon"])

            # Save reward heatmap
            self._save_reward_heatmap(
                cars_list, num_divisions, simulation_params["epsilon"]
            )

            # Save trajectory data
            self._save_trajectory_data(car_trajectories, simulation_params["epsilon"])

            # Save experience data
            self._save_experience_data(cars_list, simulation_params["epsilon"])

            # Save simulation parameters
            self._save_simulation_parameters(simulation_params)

            # Save summary statistics
            self._save_summary_statistics(cars_list, total_rewards, simulation_params)

            logger.info("All simulation results saved successfully")

        except Exception as e:
            logger.error(f"Error saving simulation results: {e}")
            raise

    def _save_reward_timeline(self, total_rewards: list[float], epsilon: float) -> None:
        """Save reward timeline data and plot."""
        # Save raw data
        reward_file = self.output_dir / f"total_reward_{epsilon}.txt"
        with reward_file.open("w") as f:
            for reward in total_rewards:
                f.write(f"{reward}\\n")

        # Create and save plot
        plot_file = self.output_dir / f"reward_timeline_{epsilon}.png"
        self.visualizer.plot_reward_timeline(
            total_rewards, plot_file, title=f"Total Reward Over Time (ε={epsilon})"
        )

        logger.debug(f"Saved reward timeline: {reward_file}, {plot_file}")

    def _save_reward_heatmap(
        self, cars_list: list, num_divisions: int, epsilon: float
    ) -> None:
        """Save reward heatmap."""
        heatmap_file = self.output_dir / f"reward_heatmap_{epsilon}.png"
        self.visualizer.create_reward_heatmap(
            cars_list,
            num_divisions,
            heatmap_file,
            title=f"Average Reward per Step by Area (ε={epsilon})",
        )

        logger.debug(f"Saved reward heatmap: {heatmap_file}")

    def _save_trajectory_data(
        self, car_trajectories: list[tuple], epsilon: float
    ) -> None:
        """Save car trajectory data."""
        trajectory_file = (
            self.output_dir / f"destination_coordinates_data_{epsilon}.txt"
        )

        with trajectory_file.open("w") as f:
            for car_id, time_step, origin_pos, dest_pos in car_trajectories:
                f.write(f"{car_id},{time_step},{origin_pos},{dest_pos}\\n")

        logger.debug(f"Saved trajectory data: {trajectory_file}")

    def _save_experience_data(self, cars_list: list, epsilon: float) -> None:
        """Save car experience data."""
        experience_file = self.output_dir / f"experience_{epsilon}.txt"

        with experience_file.open("w") as f:
            for car_idx, car in enumerate(cars_list):
                for time_slot in range(24):
                    for area_coords, experience in car.experience[time_slot].items():
                        f.write(
                            f"car_{car_idx},time_{time_slot},"
                            f"area_{area_coords[0]}_{area_coords[1]},"
                            f"reward_{experience.reward},"
                            f"count_{experience.count},"
                            f"steps_{experience.step},"
                            f"reward_per_step_{experience.reward_per_step}\\n"
                        )

        logger.debug(f"Saved experience data: {experience_file}")

    def _save_experience_data_json(self, cars_list: list, epsilon: float) -> None:
        """Save car experience data in JSON format."""
        experience_file = self.output_dir / f"experience_{epsilon}.json"

        experience_data = {}
        for car_idx, car in enumerate(cars_list):
            car_data = {}
            for time_slot in range(24):
                time_data = {}
                for area_coords, experience in car.experience[time_slot].items():
                    area_key = f"{area_coords[0]}_{area_coords[1]}"
                    time_data[area_key] = {
                        "reward": experience.reward,
                        "count": experience.count,
                        "steps": experience.step,
                        "reward_per_step": experience.reward_per_step,
                    }
                car_data[f"time_{time_slot}"] = time_data
            experience_data[f"car_{car_idx}"] = car_data

        with experience_file.open("w") as f:
            json.dump(experience_data, f, indent=2)

        logger.debug(f"Saved experience data (JSON): {experience_file}")

    def _save_simulation_parameters(self, params: dict[str, Any]) -> None:
        """Save simulation parameters."""
        params_file = self.output_dir / "simulation_parameters.json"

        # Convert Path objects to strings for JSON serialization
        serializable_params = {}
        for key, value in params.items():
            if isinstance(value, Path):
                serializable_params[key] = str(value)
            else:
                serializable_params[key] = value

        with params_file.open("w") as f:
            json.dump(serializable_params, f, indent=2)

        logger.debug(f"Saved simulation parameters: {params_file}")

    def _save_summary_statistics(
        self, cars_list: list, total_rewards: list[float], params: dict[str, Any]
    ) -> None:
        """Save summary statistics."""
        stats_file = self.output_dir / "summary_statistics.json"

        # Calculate statistics
        final_rewards = [car.total_reward for car in cars_list]

        stats = {
            "simulation_parameters": {
                "number_of_cars": params.get("number_of_cars", 0),
                "epsilon": params.get("epsilon", 0.0),
                "max_steps": params.get("max_steps", 0),
                "num_divisions": params.get("num_divisions", 0),
            },
            "reward_statistics": {
                "total_steps": len(total_rewards),
                "final_average_reward": (
                    np.mean(total_rewards[-100:])
                    if len(total_rewards) >= 100
                    else np.mean(total_rewards)
                ),
                "max_average_reward": np.max(total_rewards) if total_rewards else 0.0,
                "min_average_reward": np.min(total_rewards) if total_rewards else 0.0,
                "final_reward_std": (
                    np.std(total_rewards[-100:])
                    if len(total_rewards) >= 100
                    else np.std(total_rewards)
                ),
            },
            "car_statistics": {
                "total_cars": len(cars_list),
                "average_final_reward": (
                    np.mean(final_rewards) if final_rewards else 0.0
                ),
                "max_final_reward": np.max(final_rewards) if final_rewards else 0.0,
                "min_final_reward": np.min(final_rewards) if final_rewards else 0.0,
                "reward_std": np.std(final_rewards) if final_rewards else 0.0,
            },
            "learning_statistics": {
                "total_experiences": sum(
                    len(car.experience[t]) for car in cars_list for t in range(24)
                ),
                "average_experiences_per_car": (
                    np.mean(
                        [
                            sum(len(car.experience[t]) for t in range(24))
                            for car in cars_list
                        ]
                    )
                    if cars_list
                    else 0.0
                ),
            },
        }

        with stats_file.open("w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved summary statistics: {stats_file}")
        logger.info(
            f"Final average reward: ${stats['reward_statistics']['final_average_reward']:.2f}"
        )
        logger.info(
            f"Total simulation steps: {stats['reward_statistics']['total_steps']}"
        )

    def create_final_report(
        self,
        cars_list: list,
        total_rewards: list[float],
        simulation_params: dict[str, Any],
    ) -> None:
        """Create a final simulation report."""
        report_file = self.output_dir / "simulation_report.md"

        # Calculate key metrics
        final_rewards = [car.total_reward for car in cars_list]
        avg_final_reward = np.mean(final_rewards) if final_rewards else 0.0
        total_steps = len(total_rewards)

        report_content = f"""# Taxi Mobility Simulation Report

## Simulation Parameters
- **Number of Cars**: {simulation_params.get('number_of_cars', 'N/A')}
- **Epsilon (ε-greedy)**: {simulation_params.get('epsilon', 'N/A')}
- **Maximum Steps**: {simulation_params.get('max_steps', 'N/A')}
- **Area Divisions**: {simulation_params.get('num_divisions', 'N/A')}
- **Total Steps Completed**: {total_steps}

## Results Summary
- **Average Final Reward per Car**: ${avg_final_reward:.2f}
- **Best Performing Car**: ${np.max(final_rewards):.2f}
- **Worst Performing Car**: ${np.min(final_rewards):.2f}
- **Reward Standard Deviation**: ${np.std(final_rewards):.2f}

## Learning Progress
- **Final Average Reward**: ${np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards):.2f}
- **Peak Average Reward**: ${np.max(total_rewards) if total_rewards else 0.0:.2f}

## Files Generated
- `total_reward_{simulation_params.get('epsilon', 'X')}.txt` - Raw reward timeline data
- `reward_timeline_{simulation_params.get('epsilon', 'X')}.png` - Reward progression plot
- `reward_heatmap_{simulation_params.get('epsilon', 'X')}.png` - Spatial reward distribution
- `destination_coordinates_data_{simulation_params.get('epsilon', 'X')}.txt` - Car trajectory data
- `experience_{simulation_params.get('epsilon', 'X')}.txt` - Learning experience data
- `simulation_parameters.json` - Complete parameter set
- `summary_statistics.json` - Detailed statistics

## Notes
This simulation used reinforcement learning (ε-greedy strategy) to model taxi movement patterns
based on historical San Francisco taxi data. Cars learned to optimize their routes based on
passenger pickup probabilities and reward feedback.
"""

        with report_file.open("w") as f:
            f.write(report_content)

        logger.info(f"Created simulation report: {report_file}")
