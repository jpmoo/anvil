#!/bin/bash

# Run script for Anvil ChatBot Electron app

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 node_modules not found. Running setup..."
    ./setup.sh
fi

# Check if Electron is installed
if [ ! -d "node_modules/electron" ]; then
    echo "❌ Electron is not installed. Running setup..."
    ./setup.sh
fi

# Run the Electron app
echo "🚀 Starting Anvil ChatBot..."
npm start








