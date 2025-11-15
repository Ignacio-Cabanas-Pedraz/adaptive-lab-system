#!/bin/bash

################################################################################
# RunPod Setup Script for Adaptive Lab System
# This script sets up the complete environment on a RunPod GPU instance
################################################################################

set -e  # Exit on error

echo "========================================"
echo "Adaptive Lab System - RunPod Setup"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on GPU
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_success "GPU detected"
else
    print_error "No GPU found. This system requires a GPU."
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
apt-get update -qq

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y -qq \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

print_success "System dependencies installed"

# Check disk space
print_status "Checking disk space..."
df -h /

# Clean up space before installation
print_status "Cleaning up disk space..."
pip cache purge 2>/dev/null || true
apt-get clean
rm -rf /tmp/*
print_success "Cleanup complete"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip --no-cache-dir

# Install Python dependencies with no cache to save space
print_status "Installing Python dependencies (this may take a few minutes)..."
print_status "Using --no-cache-dir to minimize disk usage"
pip install --no-cache-dir -r requirements.txt

print_success "Python dependencies installed"

# Clean up after installation
print_status "Cleaning up post-installation..."
pip cache purge 2>/dev/null || true
apt-get clean
rm -rf /tmp/*
print_success "Post-installation cleanup complete"

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p checkpoints
mkdir -p configs/sam2.1
mkdir -p output
mkdir -p videos

print_success "Directory structure created"

# Download SAM 2 checkpoint
print_status "Downloading SAM 2 checkpoint (hiera_tiny)..."
cd checkpoints

if [ ! -f "sam2.1_hiera_tiny.pt" ]; then
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    print_success "SAM 2 checkpoint downloaded"
else
    print_status "SAM 2 checkpoint already exists, skipping download"
fi

cd ..

# Download SAM 2 config
print_status "Downloading SAM 2 config..."
cd configs/sam2.1

if [ ! -f "sam2.1_hiera_t.yaml" ]; then
    wget -q https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
    print_success "SAM 2 config downloaded"
else
    print_status "SAM 2 config already exists, skipping download"
fi

cd ../..

# Download YOLO model (will auto-download on first use, but we can pre-download)
print_status "YOLO models will be downloaded automatically on first use"

# Verify installation
print_status "Verifying installation..."

python3 << EOF
import torch
import sys

print("\nPython Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Try importing key packages
try:
    import ultralytics
    print("✓ Ultralytics (YOLO) imported successfully")
except ImportError as e:
    print("✗ Failed to import ultralytics:", e)

try:
    import sam2
    print("✓ SAM 2 imported successfully")
except ImportError as e:
    print("✗ Failed to import SAM 2:", e)

try:
    import clip
    print("✓ CLIP imported successfully")
except ImportError as e:
    print("✗ Failed to import CLIP:", e)

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print("✗ Failed to import OpenCV:", e)

EOF

print_success "Installation verification complete"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Upload your test video to the 'videos/' directory"
echo "2. Run: python process_video.py --video videos/your_video.mp4"
echo ""
echo "Example commands:"
echo "  # Process with auto mode (recommended)"
echo "  python process_video.py --video videos/test.mp4 --save-json"
echo ""
echo "  # Process in discovery mode only"
echo "  python process_video.py --video videos/test.mp4 --mode discovery"
echo ""
echo "  # Process every 5th frame (faster)"
echo "  python process_video.py --video videos/test.mp4 --skip-frames 5"
echo ""
echo "  # Process with mask visualization"
echo "  python process_video.py --video videos/test.mp4 --save-masks --save-json"
echo ""
