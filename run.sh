#!/bin/bash

# Anvil Quick Start Script

echo "🔨 Starting Anvil..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start Streamlit
echo "Starting Anvil application..."
streamlit run main.py


































