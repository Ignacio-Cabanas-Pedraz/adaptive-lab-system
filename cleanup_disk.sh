#!/bin/bash

################################################################################
# Disk Cleanup Script for RunPod
# Run this if you encounter "No space left on device" errors
################################################################################

echo "========================================"
echo "RunPod Disk Cleanup Utility"
echo "========================================"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Show disk usage before
echo "Disk usage BEFORE cleanup:"
df -h /
echo ""

# Clean pip cache
print_status "Cleaning pip cache..."
pip cache purge 2>/dev/null || true
print_success "Pip cache cleared"

# Clean apt cache
print_status "Cleaning apt cache..."
apt-get clean
apt-get autoremove -y
print_success "Apt cache cleared"

# Clean tmp files
print_status "Cleaning temporary files..."
rm -rf /tmp/*
rm -rf /var/tmp/*
print_success "Temporary files cleared"

# Clean Python cache
print_status "Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
print_success "Python cache cleared"

# Clean YOLO cache (if exists)
print_status "Cleaning YOLO cache..."
rm -rf ~/.cache/yolo 2>/dev/null || true
rm -rf runs/ 2>/dev/null || true
print_success "YOLO cache cleared"

# Clean old logs
print_status "Cleaning old logs..."
find /var/log -type f -name "*.log" -delete 2>/dev/null || true
find /var/log -type f -name "*.gz" -delete 2>/dev/null || true
print_success "Old logs cleared"

# Show disk usage after
echo ""
echo "Disk usage AFTER cleanup:"
df -h /
echo ""

print_success "Cleanup complete!"
echo ""
echo "If you still need more space:"
echo "1. Increase RunPod storage in pod settings (recommended: 50-100GB)"
echo "2. Delete unnecessary files from output/ directory"
echo "3. Remove old videos from videos/ directory"
echo "4. Use smaller models (e.g., yolov8n instead of yolov8l)"
