#!/bin/bash
# Download large data files for taxi mobility simulator

set -e

echo "🚀 Downloading large data files for taxi mobility simulator..."

# Create directories
mkdir -p data/maps
mkdir -p data/raw/cabspottingdata

# Note: These are placeholder URLs - replace with actual data sources
echo "📍 Note: Large data files (XML, OSM) should be obtained from:"
echo "   - SUMO network files: https://sumo.dlr.de/"
echo "   - OpenStreetMap data: https://www.openstreetmap.org/"
echo "   - Taxi data: San Francisco cab spotting data"

echo ""
echo "📋 Required files:"
echo "   data/maps/EntireSanFrancisco.net.xml (277MB)"
echo "   data/maps/EntireSanFrancisco.osm (406MB)"
echo "   data/raw/cabspottingdata/_cabs.txt"
echo "   data/raw/cabspottingdata/new_*.txt (taxi trajectory files)"

echo ""
echo "⚠️  Due to size constraints, these files are not included in the repository."
echo "   Please obtain them from the original sources and place them in the"
echo "   appropriate directories as shown above."

echo ""
echo "✅ Small configuration files (.geojson, .poly) are already included."