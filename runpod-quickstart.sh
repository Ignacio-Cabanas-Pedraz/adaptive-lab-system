#!/bin/bash
# RunPod Quick Setup Script for Adaptive Lab System
# Run this script on your RunPod instance after connecting

set -e

echo "=========================================="
echo "Adaptive Lab System - RunPod Quick Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on RunPod
if [ ! -d "/workspace" ]; then
    echo -e "${YELLOW}Warning: /workspace directory not found. Are you running on RunPod?${NC}"
    echo "Continuing anyway..."
fi

# Navigate to workspace
cd /workspace || cd ~

# Clone repository if not exists
if [ ! -d "adaptive-lab-system" ]; then
    echo -e "${BLUE}📦 Cloning repository...${NC}"
    git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
else
    echo -e "${GREEN}✓ Repository already exists${NC}"
    cd adaptive-lab-system
    echo -e "${BLUE}📦 Pulling latest changes...${NC}"
    git pull origin main
fi

cd adaptive-lab-system

# Update system
echo -e "${BLUE}🔄 Updating system packages...${NC}"
apt-get update > /dev/null 2>&1

# Install system dependencies
echo -e "${BLUE}📦 Installing system dependencies...${NC}"
apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    wget > /dev/null 2>&1

echo -e "${GREEN}✓ System dependencies installed${NC}"

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
echo "   This may take a few minutes..."
pip install -r requirements.txt > /dev/null 2>&1
pip install -r backend/requirements.txt > /dev/null 2>&1

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p videos weights results logs backend/uploads backend/templates

# Check GPU
echo -e "${BLUE}🎮 Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected. Llama will run on CPU (slower)${NC}"
fi

# Check CUDA
echo -e "${BLUE}🔥 Checking CUDA...${NC}"
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ CUDA not available")
EOF

# Check HuggingFace authentication
echo -e "${BLUE}🤗 Checking HuggingFace authentication...${NC}"
if huggingface-cli whoami > /dev/null 2>&1; then
    USERNAME=$(huggingface-cli whoami 2>&1 | tail -n 1)
    echo -e "${GREEN}✓ Logged in to HuggingFace as: ${USERNAME}${NC}"
else
    echo -e "${YELLOW}⚠ Not logged in to HuggingFace${NC}"
    echo "  Run: huggingface-cli login"
    echo "  Or: export HF_TOKEN='your_token'"
fi

# Verify setup
echo -e "${BLUE}🔍 Verifying installation...${NC}"
if python scripts/verify_setup.py; then
    echo -e "${GREEN}✓ Setup verification passed${NC}"
else
    echo -e "${YELLOW}⚠ Some components may not be fully configured${NC}"
fi

# Test parameter extraction
echo -e "${BLUE}🧪 Testing parameter extraction...${NC}"
python test_parameter_extraction.py | tail -n 5

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend:"
echo "   cd backend && python main.py"
echo ""
echo "2. Access API docs:"
echo "   http://localhost:8000/docs"
echo ""
echo "3. Test template generation:"
echo "   curl -X POST http://localhost:8000/api/upload -F 'file=@tests/fixtures/sample_procedures/DNA_Extraction.txt'"
echo ""
echo "4. (Optional) Request Llama 3 access:"
echo "   Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
echo ""
echo "5. Process videos:"
echo "   python scripts/test_video_with_tep.py --template templates/your-template.json --video videos/your-video.mp4"
echo ""
echo "For more details, see:"
echo "  - RUNPOD_SETUP.md - Complete RunPod setup guide"
echo "  - HUGGINGFACE_ACCESS.md - HuggingFace Llama 3 access"
echo "  - README.md - General documentation"
echo ""
