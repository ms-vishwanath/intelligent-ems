#!/bin/bash

# Intelligent EMS Setup Script

set -e

echo "🚑 Intelligent Emergency Response System - Setup"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Train ML model
echo "Training ML model..."
python ml/train.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API:"
echo "  source venv/bin/activate"
echo "  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Or use Docker Compose:"
echo "  cd infra && docker-compose up"

