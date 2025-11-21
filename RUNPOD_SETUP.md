# RunPod Setup Guide for Adaptive Lab System

Complete guide for deploying the Adaptive Lab System on RunPod with GPU support.

## Prerequisites

1. ✅ Code pushed to GitHub: https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system
2. ⏳ HuggingFace Llama 3 access (optional but recommended)
3. RunPod account with credit

## Part 1: Create RunPod Pod

### 1.1 Choose Template

1. Log into [RunPod.io](https://runpod.io)
2. Click **"Deploy"** → **"Pods"**
3. Select a template:
   - **Recommended:** "PyTorch 2.1" or "RunPod Pytorch"
   - **GPU:** Choose based on budget:
     - RTX 4090 (24GB VRAM) - Best performance
     - RTX 3090 (24GB VRAM) - Good balance
     - RTX A5000 (24GB VRAM) - Professional option
     - A40 (48GB VRAM) - If you need more VRAM

### 1.2 Configure Pod

**Container Settings:**
```
Container Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
Container Disk: 50 GB (minimum)
Volume Disk: 100 GB (recommended for videos)
```

**Exposed Ports:**
```
8000 (Backend API)
3000 (Frontend - optional if using)
```

**Environment Variables:**
```bash
HF_TOKEN=<your_huggingface_token>
GITHUB_REPO=https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system
```

### 1.3 Start Pod

Click **"Deploy On-Demand"** or **"Deploy Spot"** (cheaper but can be interrupted)

## Part 2: Initial Setup on RunPod

### 2.1 Connect to Pod

```bash
# Via SSH (recommended)
ssh root@<pod-ip> -p <ssh-port>

# Or use RunPod Web Terminal
# Click "Connect" → "Start Web Terminal"
```

### 2.2 Clone Repository

```bash
# Clone your repository
cd /workspace
git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
cd adaptive-lab-system
```

### 2.3 Install System Dependencies

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Install Node.js and npm (for frontend)
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs
```

### 2.4 Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install backend dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install HuggingFace CLI
pip install huggingface-hub
```

### 2.5 Authenticate with HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token directly
export HF_TOKEN="your_token_here"
echo "export HF_TOKEN='your_token_here'" >> ~/.bashrc
```

## Part 3: Configure and Test

### 3.1 Verify Setup

```bash
# Run verification script
python scripts/verify_setup.py

# Test parameter extraction
python test_parameter_extraction.py

# Test Llama access (if you have access)
python scripts/test_llama_integration.py
```

### 3.2 Test Backend Locally

```bash
# Start backend
cd backend
python main.py

# Test API (in another terminal)
curl http://localhost:8000/health
```

### 3.3 Create Systemd Service (Production)

```bash
# Create backend service
cat > /etc/systemd/system/adaptive-lab-backend.service << 'EOF'
[Unit]
Description=Adaptive Lab System Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/adaptive-lab-system/backend
Environment="HF_TOKEN=your_token_here"
Environment="PYTHONPATH=/workspace/adaptive-lab-system"
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable adaptive-lab-backend
systemctl start adaptive-lab-backend
systemctl status adaptive-lab-backend
```

## Part 4: Frontend Setup (Optional)

### 4.1 Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 4.2 Configure Environment

```bash
# Create .env file
cat > .env << EOF
VITE_API_URL=http://<your-runpod-ip>:8000
EOF
```

### 4.3 Build Frontend

```bash
# Development mode
npm run dev

# Production build
npm run build
```

### 4.4 Serve Frontend (Production)

```bash
# Install serve
npm install -g serve

# Serve build
serve -s dist -p 3000

# Or use nginx
apt-get install -y nginx
cp -r dist/* /var/www/html/
systemctl restart nginx
```

## Part 5: Video Processing Setup

### 5.1 Download YOLO Weights

```bash
# Create weights directory
mkdir -p weights

# Download YOLOv8 weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O weights/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O weights/yolov8s.pt
```

### 5.2 Test Video Processing

```bash
# Upload a test video to /workspace/adaptive-lab-system/videos/

# Process with template
python scripts/test_video_with_tep.py \
  --template templates/your-template.json \
  --video videos/your-video.mp4 \
  --output results/
```

## Part 6: Usage

### 6.1 Create Templates via API

```bash
# Upload PDF/text file
curl -X POST http://localhost:8000/api/upload \
  -F "file=@procedure.pdf"

# Generate template
curl -X POST http://localhost:8000/api/generate-template \
  -H "Content-Type: application/json" \
  -d '{
    "title": "DNA Extraction",
    "steps": ["Step 1...", "Step 2..."],
    "user_id": "researcher_1"
  }'
```

### 6.2 Access Frontend

Open browser and navigate to:
```
http://<your-runpod-ip>:3000
```

### 6.3 API Documentation

Access FastAPI docs at:
```
http://<your-runpod-ip>:8000/docs
```

## Part 7: Monitoring and Maintenance

### 7.1 View Logs

```bash
# Backend logs
journalctl -u adaptive-lab-backend -f

# Or if running manually
tail -f backend.log
```

### 7.2 GPU Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check CUDA availability
python << EOF
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA version: {torch.version.cuda}")
EOF
```

### 7.3 Update Code

```bash
# Pull latest changes
cd /workspace/adaptive-lab-system
git pull origin main

# Reinstall dependencies if needed
pip install -r requirements.txt

# Restart services
systemctl restart adaptive-lab-backend
```

## Part 8: Optimization Tips

### 8.1 Llama Model Caching

The system automatically caches Llama models. First load takes ~2-3 minutes, subsequent loads are instant.

### 8.2 Batch Processing

For multiple videos, use batch processing:

```python
from src.integration.procedure_executor import ProcedureExecutor

executor = ProcedureExecutor("templates/template.json")

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
for video in videos:
    results = executor.process_video(video)
```

### 8.3 Resource Management

```bash
# Clear Llama cache if needed
python << EOF
import torch
torch.cuda.empty_cache()
EOF

# Monitor disk space
df -h

# Clean up old results
rm -rf results/old_*
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Use a smaller batch size or upgrade to a GPU with more VRAM

### Issue: "Cannot access gated repo" (Llama)
**Solution:** Request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

### Issue: Port not accessible
**Solution:**
1. Check RunPod port forwarding settings
2. Ensure pod firewall allows connections
3. Use RunPod's proxy URL instead of direct IP

### Issue: Slow performance
**Solution:**
1. Check GPU utilization with `nvidia-smi`
2. Ensure CUDA is being used (not CPU)
3. Consider upgrading to a faster GPU

## Cost Optimization

1. **Use Spot Instances:** 3-5x cheaper but can be interrupted
2. **Stop Pod When Idle:** Only pay for active time
3. **Use Smaller GPUs for Testing:** Switch to larger GPUs for production
4. **Process Videos in Batches:** Maximize GPU utilization

## Next Steps

1. ✅ Set up pod and clone repository
2. ✅ Install dependencies and verify setup
3. ✅ Request Llama 3 access on HuggingFace
4. ✅ Test template generation
5. ✅ Process test videos
6. 🚀 Deploy to production!

## Quick Start Script

Save and run this script to automate most of the setup:

```bash
#!/bin/bash
# quick-setup.sh

set -e

echo "🚀 Setting up Adaptive Lab System on RunPod..."

# Clone repo
cd /workspace
if [ ! -d "adaptive-lab-system" ]; then
    git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
fi
cd adaptive-lab-system

# Install system dependencies
apt-get update
apt-get install -y python3-pip ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Verify setup
python scripts/verify_setup.py

# Test parameter extraction
python test_parameter_extraction.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Configure HuggingFace: huggingface-cli login"
echo "2. Start backend: cd backend && python main.py"
echo "3. Access API docs: http://localhost:8000/docs"
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system/issues
- RunPod Docs: https://docs.runpod.io/
- HuggingFace Docs: https://huggingface.co/docs
