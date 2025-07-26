"""Visualization utilities for the taxi mobility simulator."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class Visualizer:
    """Handle visualization and plotting for the simulation."""

    def __init__(self, dpi: int = 300, style: str = "whitegrid"):
        """Initialize visualizer with plotting parameters."""
        self.dpi = dpi
        plt.style.use("default")
        sns.set_style(style)

    def create_probability_heatmaps(
        self,
        ride_probabilities: list[list[list[float]]],
        output_dir: Path,
        prefix: str = "ride_probability",
    ) -> None:
        """Create heatmaps for ride probabilities by time slot."""
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Creating probability heatmaps for 24 time slots")

        for hour in range(24):
            try:
                fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)

                # Create heatmap
                sns.heatmap(
                    ride_probabilities[hour],
                    cmap="coolwarm",
                    square=True,
                    robust=True,
                    ax=ax,
                    cbar_kws={"label": "Ride Probability"},
                )

                # Customize plot
                ax.set_title(
                    f"Ride Probability Distribution - Hour {hour:02d}:00", fontsize=14
                )
                ax.set_xlabel("X Grid", fontsize=12)
                ax.set_ylabel("Y Grid", fontsize=12)

                # Remove tick labels for cleaner look
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(length=0)

                # Save plot
                output_file = output_dir / f"{prefix}_{hour:02d}.png"
                plt.savefig(output_file, bbox_inches="tight", dpi=self.dpi)
                plt.close(fig)

                logger.debug(f"Saved heatmap for hour {hour} to {output_file}")

            except Exception as e:
                logger.error(f"Error creating heatmap for hour {hour}: {e}")
                continue

        logger.info(f"Completed creating heatmaps in {output_dir}")

    def create_reward_heatmap(
        self,
        cars_list: list,
        num_divisions: int,
        output_file: Path,
        title: str = "Average Reward per Step by Area",
    ) -> None:
        """Create heatmap showing average reward per step for each area."""
        try:
            # Initialize reward and step tracking
            total_rewards = np.zeros((num_divisions, num_divisions))
            total_steps = np.zeros((num_divisions, num_divisions))

            # Aggregate data from all cars
            for car in cars_list:
                for time_slot in range(24):
                    for (x, y), experience in car.experience[time_slot].items():
                        if 0 <= x < num_divisions and 0 <= y < num_divisions:
                            total_rewards[y, x] += experience.reward
                            total_steps[y, x] += experience.step

            # Calculate average reward per step
            reward_per_step = np.zeros((num_divisions, num_divisions))
            mask = total_steps > 0
            reward_per_step[mask] = total_rewards[mask] / total_steps[mask]

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)

            sns.heatmap(
                reward_per_step,
                cmap="Blues",
                square=True,
                ax=ax,
                cbar_kws={"label": "Reward per Step ($)"},
            )

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("X Grid", fontsize=12)
            ax.set_ylabel("Y Grid", fontsize=12)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)

            # Save plot
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)

            logger.info(f"Saved reward heatmap to {output_file}")

        except Exception as e:
            logger.error(f"Error creating reward heatmap: {e}")
            raise

    def plot_reward_timeline(
        self,
        total_rewards: list[float],
        output_file: Path,
        title: str = "Total Reward Over Time",
        window_size: int = 100,
    ) -> None:
        """Plot the evolution of total rewards over simulation time."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)

            steps = list(range(len(total_rewards)))

            # Plot raw data
            ax.plot(
                steps, total_rewards, alpha=0.3, color="lightblue", label="Raw Data"
            )

            # Plot moving average if enough data points
            if len(total_rewards) > window_size:
                moving_avg = self._calculate_moving_average(total_rewards, window_size)
                ax.plot(
                    steps[window_size - 1 :],
                    moving_avg,
                    color="darkblue",
                    linewidth=2,
                    label=f"Moving Average ({window_size} steps)",
                )

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Simulation Step", fontsize=12)
            ax.set_ylabel("Average Reward per Car ($)", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Save plot
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)

            logger.info(f"Saved reward timeline to {output_file}")

        except Exception as e:
            logger.error(f"Error creating reward timeline: {e}")
            raise

    def create_car_distribution_plot(
        self,
        car_positions: list[tuple],
        ride_states: list[bool],
        destinations: list[tuple],
        map_bounds: tuple,
        output_file: Path,
        title: str = "Car Distribution",
    ) -> None:
        """Create a scatter plot showing car positions and destinations."""
        try:
            fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

            # Separate empty and occupied cars
            empty_cars = [
                (x, y)
                for (x, y), occupied in zip(car_positions, ride_states, strict=False)
                if not occupied
            ]
            occupied_cars = [
                (x, y)
                for (x, y), occupied in zip(car_positions, ride_states, strict=False)
                if occupied
            ]
            empty_dests = [
                (x, y)
                for (x, y), occupied in zip(destinations, ride_states, strict=False)
                if not occupied
            ]
            occupied_dests = [
                (x, y)
                for (x, y), occupied in zip(destinations, ride_states, strict=False)
                if occupied
            ]

            # Plot cars
            if empty_cars:
                empty_x, empty_y = zip(*empty_cars, strict=False)
                ax.scatter(
                    empty_x,
                    empty_y,
                    c="green",
                    marker="s",
                    s=25,
                    alpha=0.7,
                    label="Empty Cars",
                )

            if occupied_cars:
                occupied_x, occupied_y = zip(*occupied_cars, strict=False)
                ax.scatter(
                    occupied_x,
                    occupied_y,
                    c="blue",
                    marker="s",
                    s=25,
                    alpha=0.7,
                    label="Occupied Cars",
                )

            # Plot destinations
            if empty_dests:
                dest_x, dest_y = zip(*empty_dests, strict=False)
                ax.scatter(
                    dest_x,
                    dest_y,
                    c="red",
                    marker="*",
                    s=30,
                    alpha=0.6,
                    label="Empty Car Destinations",
                )

            if occupied_dests:
                dest_x, dest_y = zip(*occupied_dests, strict=False)
                ax.scatter(
                    dest_x,
                    dest_y,
                    c="blue",
                    marker="*",
                    s=30,
                    alpha=0.6,
                    label="Occupied Car Destinations",
                )

            # Set map bounds
            x0, x1, y0, y1 = map_bounds
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.invert_yaxis()  # Match image coordinate system

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("X Coordinate", fontsize=12)
            ax.set_ylabel("Y Coordinate", fontsize=12)
            ax.legend()

            # Save plot
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)

            logger.debug(f"Saved car distribution plot to {output_file}")

        except Exception as e:
            logger.error(f"Error creating car distribution plot: {e}")
            raise

    def _calculate_moving_average(
        self, data: list[float], window_size: int
    ) -> list[float]:
        """Calculate moving average of data."""
        if len(data) < window_size:
            return data

        moving_avg = []
        for i in range(window_size - 1, len(data)):
            window = data[i - window_size + 1 : i + 1]
            moving_avg.append(sum(window) / window_size)

        return moving_avg
