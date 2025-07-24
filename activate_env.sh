#!/bin/bash
# Script to activate the drill-assistant virtual environment

echo "🔧 Activating drill-assistant virtual environment..."
source ~/.virtualenvs/drill-assistant/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated successfully!"
    echo "🐍 Python version: $(python --version)"
    echo "📦 Virtual environment: $(which python)"
    echo ""
    echo "🚀 Ready to run the oil & gas data processing pipeline!"
    echo "   python main.py                    # Run complete pipeline"
    echo "   python main.py --interactive      # Interactive mode"
    echo ""
    echo "💡 To deactivate: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi 