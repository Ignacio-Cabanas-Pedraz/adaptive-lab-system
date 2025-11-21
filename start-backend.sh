#!/bin/bash

echo "🚀 Starting FastAPI Backend..."
echo "================================"
echo ""
echo "Make sure you have installed dependencies:"
echo "  pip install -r backend/requirements.txt"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
python main.py
