#!/bin/bash

echo "🚀 Starting React Frontend..."
echo "============================="
echo ""
echo "Make sure you have installed dependencies:"
echo "  cd frontend && npm install"
echo ""
echo "Frontend will be available at: http://localhost:3000"
echo "Backend must be running on port 8000!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd frontend
npm run dev
