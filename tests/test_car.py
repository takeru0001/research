"""Tests for the Car model."""

import networkx as nx
import pytest

from src.taxi_mobility_simulator.models.car import Car, ExperienceData, RideState


class TestCar:
    """Test cases for the Car class."""

    def test_car_initialization(self):
        """Test car initialization with valid parameters."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        assert car.orig_node_id == 1
        assert car.dest_node_id == 2
        assert car.shortest_path == [1, 3, 2]
        assert car.num_of_division == 10
        assert len(car.experience) == 24
        assert car.total_reward == 0.0
        assert car.goal_arrived is False
        assert car.ride_flag is False

    def test_car_init_with_graph(self):
        """Test car initialization with a graph."""
        # Create a simple graph
        graph = nx.DiGraph()
        graph.add_node(1, pos=(0, 0))
        graph.add_node(2, pos=(10, 0))
        graph.add_node(3, pos=(5, 5))
        graph.add_edge(1, 3, weight=7.07, speed=30)
        graph.add_edge(3, 2, weight=7.07, speed=30)

        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        car.init(graph)

        assert car.current_start_node == [0, 0]
        assert car.current_end_node == [5, 5]
        assert car.current_position == [0, 0]
        assert car.current_distance == 7.07

    def test_car_init_invalid_path(self):
        """Test car initialization with invalid path."""
        graph = nx.DiGraph()
        graph.add_node(1, pos=(0, 0))

        car = Car(
            orig_node_id=1,
            dest_node_id=2,
            shortest_path=[1],  # Too short
            num_of_division=10,
        )

        with pytest.raises(
            ValueError, match="Shortest path must have at least 2 nodes"
        ):
            car.init(graph)

    def test_update_experience(self):
        """Test experience update functionality."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        # Update experience
        car.num_of_elapsed_steps = 10
        car.update_experience((5, 5), 12, 15.0)

        assert (5, 5) in car.experience[12]
        exp_data = car.experience[12][(5, 5)]
        assert exp_data.reward == 15.0
        assert exp_data.count == 1
        assert exp_data.step == 10
        assert exp_data.reward_per_step == 1.5
        assert car.total_reward == 15.0

    def test_update_experience_invalid_time(self):
        """Test experience update with invalid time slot."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        # Should not raise exception, but should log warning
        car.update_experience((5, 5), 25, 15.0)  # Invalid time slot

        # Experience should not be updated
        assert (5, 5) not in car.experience[23]
        assert car.total_reward == 0.0

    def test_get_best_area(self):
        """Test getting the best area based on experience."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        # Add some experience data
        car.experience[10][(1, 1)] = ExperienceData(
            reward=10.0, step=5, reward_per_step=2.0
        )
        car.experience[10][(2, 2)] = ExperienceData(
            reward=15.0, step=3, reward_per_step=5.0
        )
        car.experience[10][(3, 3)] = ExperienceData(
            reward=8.0, step=4, reward_per_step=2.0
        )

        best_area = car.get_best_area(10)
        assert best_area == (2, 2)  # Highest reward per step

    def test_get_best_area_no_experience(self):
        """Test getting best area when no experience exists."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        best_area = car.get_best_area(10)
        assert best_area is None

    def test_get_best_area_invalid_time(self):
        """Test getting best area with invalid time slot."""
        car = Car(
            orig_node_id=1, dest_node_id=2, shortest_path=[1, 3, 2], num_of_division=10
        )

        best_area = car.get_best_area(25)  # Invalid time
        assert best_area is None


class TestExperienceData:
    """Test cases for the ExperienceData class."""

    def test_experience_data_initialization(self):
        """Test ExperienceData initialization."""
        exp_data = ExperienceData()

        assert exp_data.reward == 0.0
        assert exp_data.count == 0
        assert exp_data.step == 0
        assert exp_data.reward_per_step == 0.0

    def test_experience_data_with_values(self):
        """Test ExperienceData initialization with values."""
        exp_data = ExperienceData(reward=100.0, count=5, step=50, reward_per_step=2.0)

        assert exp_data.reward == 100.0
        assert exp_data.count == 5
        assert exp_data.step == 50
        assert exp_data.reward_per_step == 2.0


class TestRideState:
    """Test cases for the RideState enum."""

    def test_ride_state_values(self):
        """Test RideState enum values."""
        assert RideState.EMPTY.value == 0
        assert RideState.OCCUPIED.value == 1

    def test_ride_state_comparison(self):
        """Test RideState comparison."""
        assert RideState.EMPTY != RideState.OCCUPIED
        assert RideState.EMPTY == RideState.EMPTY
