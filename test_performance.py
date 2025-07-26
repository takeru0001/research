#!/usr/bin/env python3
"""Performance test script for taxi mobility simulator."""

import logging
import time

from src.taxi_mobility_simulator.config.settings import SimulationSettings
from src.taxi_mobility_simulator.simulation.simulator import TaxiSimulator
from src.taxi_mobility_simulator.utils.network_cache import SimulationCache

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_performance():
    """Test simulator performance with different configurations."""

    print("=== Performance Test: Taxi Mobility Simulator ===\n")

    # Clear cache first to ensure clean test
    cache = SimulationCache()
    cache.clear_cache()
    print("Cleared all cache files for clean test\n")

    # Small test configuration
    settings = SimulationSettings(
        number_of_cars=10,  # Small number for quick test
        max_steps=1000,  # Short simulation
        num_of_division=5,  # Smaller grid
        save_animation=False,  # Skip animation for speed
        epsilon=0.1,
    )

    print("Test Configuration:")
    print(f"  Cars: {settings.number_of_cars}")
    print(f"  Steps: {settings.max_steps}")
    print(f"  Grid divisions: {settings.num_of_division}")
    print(f"  Animation: {settings.save_animation}")
    print()

    # Store timing results
    results = {}

    # First run (no cache)
    print("=== First Run (Building from Scratch) ===")
    start_time = time.time()

    try:
        simulator = TaxiSimulator(settings)
        setup_start = time.time()
        simulator._setup_simulation()
        setup_time = time.time() - setup_start

        print(f"Setup completed in {setup_time:.2f} seconds")
        print(
            f"  Network: {simulator.road_graph.number_of_nodes()} nodes, {simulator.road_graph.number_of_edges()} edges"
        )
        print(f"  Cars initialized: {len(simulator.cars_list)}")

        # Run a few simulation steps to test
        sim_start = time.time()
        for step in range(min(100, settings.max_steps)):
            simulator._simulate_step(step)
            if step % 20 == 0:
                print(f"  Step {step}: {len(simulator.cars_list)} cars active")

        sim_time = time.time() - sim_start
        total_time = time.time() - start_time

        print(f"Simulation (100 steps) completed in {sim_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")

        results["first_run"] = {
            "setup_time": setup_time,
            "sim_time": sim_time,
            "total_time": total_time,
        }

    except Exception as e:
        print(f"Error in first run: {e}")
        return

    # Show cache info
    cache_info = cache.get_cache_info()
    print(
        f"\nCache created: {cache_info.get('total_files', 0)} files, {cache_info.get('total_size_mb', 0):.2f} MB"
    )

    print("\n" + "=" * 60 + "\n")

    # Second run (with cache)
    print("=== Second Run (Using Cache) ===")
    start_time = time.time()

    try:
        simulator2 = TaxiSimulator(settings)
        setup_start = time.time()
        simulator2._setup_simulation()
        setup_time = time.time() - setup_start

        print(f"Setup completed in {setup_time:.2f} seconds")
        print(
            f"  Network: {simulator2.road_graph.number_of_nodes()} nodes, {simulator2.road_graph.number_of_edges()} edges"
        )
        print(f"  Cars initialized: {len(simulator2.cars_list)}")

        # Run a few simulation steps
        sim_start = time.time()
        for step in range(min(100, settings.max_steps)):
            simulator2._simulate_step(step)
            if step % 20 == 0:
                print(f"  Step {step}: {len(simulator2.cars_list)} cars active")

        sim_time = time.time() - sim_start
        total_time = time.time() - start_time

        print(f"Simulation (100 steps) completed in {sim_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")

        results["second_run"] = {
            "setup_time": setup_time,
            "sim_time": sim_time,
            "total_time": total_time,
        }

    except Exception as e:
        print(f"Error in second run: {e}")
        return

    print("\n" + "=" * 60 + "\n")

    # Performance comparison
    print("=== Performance Comparison ===")
    if "first_run" in results and "second_run" in results:
        first = results["first_run"]
        second = results["second_run"]

        setup_improvement = (
            (first["setup_time"] - second["setup_time"]) / first["setup_time"]
        ) * 100
        sim_improvement = (
            (first["sim_time"] - second["sim_time"]) / first["sim_time"]
        ) * 100
        total_improvement = (
            (first["total_time"] - second["total_time"]) / first["total_time"]
        ) * 100

        print(
            f"Setup time improvement: {setup_improvement:.1f}% ({first['setup_time']:.2f}s → {second['setup_time']:.2f}s)"
        )
        print(
            f"Simulation time improvement: {sim_improvement:.1f}% ({first['sim_time']:.2f}s → {second['sim_time']:.2f}s)"
        )
        print(
            f"Total time improvement: {total_improvement:.1f}% ({first['total_time']:.2f}s → {second['total_time']:.2f}s)"
        )

        if total_improvement > 50:
            print("\n🎉 Excellent! Cache system is working very effectively!")
        elif total_improvement > 20:
            print("\n✅ Good! Cache system is providing significant improvements!")
        elif total_improvement > 5:
            print("\n👍 Cache system is working, but there's room for improvement.")
        else:
            print("\n⚠️  Cache system may need optimization.")

    # Final cache info
    final_cache_info = cache.get_cache_info()
    print(
        f"\nFinal cache state: {final_cache_info.get('total_files', 0)} files, {final_cache_info.get('total_size_mb', 0):.2f} MB"
    )

    cache_types = final_cache_info.get("cache_types", {})
    if cache_types:
        print("Cache breakdown:")
        for cache_type, data in cache_types.items():
            size_mb = data["size"] / (1024 * 1024)
            print(f"  {cache_type}: {data['count']} files, {size_mb:.2f} MB")


if __name__ == "__main__":
    test_performance()
