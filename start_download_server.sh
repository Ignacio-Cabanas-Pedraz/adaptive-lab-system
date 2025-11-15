#!/bin/bash

################################################################################
# HTTP Download Server for RunPod
# Start a web server to download output files
################################################################################

echo "========================================"
echo "Starting Download Server"
echo "========================================"
echo ""

# Navigate to output directory
cd /adaptive-lab-system/output

# Check if there are any results
if [ -z "$(ls -A)" ]; then
    echo "⚠️  No output files found in /adaptive-lab-system/output/"
    echo ""
    echo "Please run the video processing first:"
    echo "  python process_video.py --video videos/your-video.mp4 --save-json"
    echo ""
    exit 1
fi

# List available results
echo "Available result folders:"
ls -lh
echo ""

# Get port (default 8000, but can be overridden)
PORT=${1:-8000}

echo "Starting HTTP server on port $PORT..."
echo ""
echo "========================================"
echo "Access your files via RunPod's web interface:"
echo ""
echo "1. Go to your RunPod Pod page"
echo "2. Click 'Connect' button"
echo "3. Look for 'HTTP Service [Port $PORT]'"
echo "4. Click the generated URL (e.g., https://xxxxx-$PORT.proxy.runpod.net)"
echo "5. Browse and download your files"
echo ""
echo "OR if you see 'TCP Port Mappings':"
echo "  Find port $PORT and click the corresponding link"
echo ""
echo "========================================"
echo ""
echo "Server is running... Press Ctrl+C to stop"
echo ""

# Start Python HTTP server
python3 -m http.server $PORT
