#!/bin/bash
# Setup script for Hyperliquid ML Trading Bot

echo "=========================================="
echo "Hyperliquid ML Trading Bot - Setup"
echo "=========================================="
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    exit 1
fi

echo "[1/4] Python 3 found: $(python3 --version)"

# Check pip3
if ! command -v pip3 &> /dev/null; then
    echo "[ERROR] pip3 is not installed"
    exit 1
fi

echo "[2/4] pip3 found"

# Install dependencies
echo "[3/4] Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo "[4/5] Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p logs

# Copy .env.example if .env doesn't exist
if [ ! -f config/.env ]; then
    echo ""
    echo "Copying config/.env.example to config/.env"
    cp config/.env.example config/.env
    echo "[WARNING] Please edit config/.env and add your API credentials"
fi

echo "[5/5] Collecting initial data for all coins..."
echo "This may take a few minutes..."
python3 collect_initial_data.py

if [ $? -ne 0 ]; then
    echo "[WARNING] Some coins failed to collect data, but continuing..."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit config/.env and add your Hyperliquid API credentials"
echo "2. Review config/settings.yaml and customize as needed"
echo "3. Train models: python3 main.py --train-only"
echo "4. Start trading (testnet): python3 main.py"
echo ""
echo "Note: Initial data has been collected for all coins."
echo "The bot will automatically update data during operation."
echo ""
