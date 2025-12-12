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

# Check if Ollama is running
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Warning: Ollama not found in PATH"
    echo "Please install Ollama from https://ollama.ai"
    echo ""
fi

# Start Streamlit
echo "Starting Anvil application..."
streamlit run main.py
















