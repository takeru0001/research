# Taxi Mobility Simulator

A modern, refactored taxi mobility simulation using reinforcement learning to model human movement patterns based on San Francisco taxi data.

![simulator](https://user-images.githubusercontent.com/58085267/142912124-a956c261-7140-44d1-98ff-ec65b0d1090d.gif)

■ taxi 　　★ taxi customer

## Overview

This simulator uses reinforcement learning to create movement models of human mobility patterns, specifically focusing on "Returner" (travelers who visit frequently visited places) and "Explorer" (travelers who visit new places) behaviors from taxi movement data.

The simulation uses real San Francisco taxi data to generate probability distributions and reward models, then applies ε-greedy reinforcement learning for optimal taxi routing decisions.

## Features

- **Modern Python Architecture**: Refactored with type hints, dataclasses, and modern Python practices
- **Configurable Settings**: Environment-based configuration with Pydantic validation
- **Comprehensive Logging**: Structured logging with multiple output formats
- **Rich Visualization**: Heatmaps, timeline plots, and spatial distribution analysis
- **CLI Interface**: Easy-to-use command-line interface with Click
- **Extensible Design**: Modular architecture for easy extension and testing
- **Data Analysis Tools**: Built-in tools for analyzing taxi data and generating probability maps

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd taxi-mobility-simulator

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd taxi-mobility-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Data

**Download large data files:**
```bash
# Run the data download script for instructions
./scripts/download_data.sh
```

**Required data files:**
- `data/maps/EntireSanFrancisco.net.xml` - SUMO network file (277MB)
- `data/maps/EntireSanFrancisco.geojson` - Geographic boundaries (included)
- `data/raw/cabspottingdata/` - Taxi data files

**Note:** Large files (XML, OSM) are not included in the repository due to size constraints. Small configuration files (.geojson, .poly) are included.

### 2. Run Simulation

```bash
# Basic simulation with default parameters
taxi-sim run

# Custom parameters
taxi-sim run --cars 200 --epsilon 0.2 --steps 100000 --divisions 15

# With configuration file
taxi-sim run --config config/custom.env
```

### 3. Analyze Data

```bash
# Generate probability maps from taxi data
taxi-sim analyze-data \
    --network-file data/maps/EntireSanFrancisco.net.xml \
    --taxi-data-dir data/raw/cabspottingdata \
    --divisions 10
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Key configuration options:
- `TAXI_SIM_NUMBER_OF_CARS`: Number of taxis in simulation
- `TAXI_SIM_EPSILON`: ε-greedy exploration rate
- `TAXI_SIM_MAX_STEPS`: Maximum simulation steps
- `TAXI_SIM_NUM_OF_DIVISION`: Grid divisions for area analysis

## Project Structure

```
taxi-mobility-simulator/
├── src/taxi_mobility_simulator/     # Main package
│   ├── models/                      # Data models (Car, Lane, etc.)
│   ├── simulation/                  # Simulation engine
│   ├── data/                        # Data processing
│   ├── utils/                       # Utilities (visualization, output)
│   ├── config/                      # Configuration management
│   └── cli.py                       # Command-line interface
├── data/                            # Data files
│   ├── raw/                         # Raw taxi data
│   ├── processed/                   # Processed data
│   └── maps/                        # Map files
├── tests/                           # Test suite
├── config/                          # Configuration files
└── output/                          # Simulation results
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run type checking
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/ --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/taxi_mobility_simulator

# Run specific test file
pytest tests/test_car.py
```

## Output

The simulation generates comprehensive output:

- **Visualizations**: Heatmaps, timeline plots, spatial distributions
- **Data Files**: Reward timelines, trajectory data, experience logs
- **Reports**: Summary statistics and final simulation report
- **Animation**: MP4 video of simulation (optional)

## Research Background

Human mobility patterns can be classified into:

- **Returner**: Travelers who primarily visit frequently visited locations
- **Explorer**: Travelers who venture to new, previously unvisited areas

This simulation models these behaviors using:
- Real San Francisco taxi data for probability distributions
- ε-greedy reinforcement learning for decision making
- Spatial-temporal reward modeling
- Network-based route optimization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`pytest`, `black`, `ruff`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{taxi_mobility_simulator,
  title={Taxi Mobility Simulator: Reinforcement Learning Based Movement Model},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/taxi-mobility-simulator}
}
```
