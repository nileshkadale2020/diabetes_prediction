#!/bin/bash
# Script to activate the dtsc691 virtual environment

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source dtsc691/bin/activate

echo "Virtual environment 'dtsc691' activated!"
echo "Python version: $(python --version)"
echo ""
echo "To deactivate, run: deactivate"
echo ""

