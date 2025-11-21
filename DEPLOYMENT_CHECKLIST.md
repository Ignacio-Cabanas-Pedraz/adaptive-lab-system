# Deployment Checklist for Adaptive Lab System

Complete checklist for deploying the system to RunPod and setting up all components.

## 📋 Pre-Deployment Checklist

### GitHub Setup ✅
- [x] Code pushed to GitHub: https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system
- [x] SSH keys configured
- [x] Repository is public and accessible

### HuggingFace Setup
- [x] Logged in as: **thatbiologicalprogrammer**
- [ ] **TODO:** Request Llama 3 access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- [ ] **TODO:** Wait for Meta's approval (usually 1-2 days)
- [ ] **TODO:** Test access after approval

### Local Testing ✅
- [x] Backend running on port 8000
- [x] Frontend running (Vite dev server)
- [x] Parameter extraction working correctly
- [x] Template generation functional
- [x] All regex improvements implemented

## 🚀 RunPod Deployment Steps

### Step 1: Create RunPod Pod

1. **Log into RunPod:** https://runpod.io
2. **Choose GPU:**
   - Recommended: RTX 4090 (24GB) or RTX 3090 (24GB)
   - Budget: RTX A5000 (24GB)
   - High memory: A40 (48GB)

3. **Select Template:**
   - PyTorch 2.1 or RunPod Pytorch

4. **Configure:**
   ```
   Container Disk: 50 GB
   Volume Disk: 100 GB
   Exposed Ports: 8000, 3000
   ```

5. **Environment Variables:**
   ```bash
   HF_TOKEN=<your_huggingface_token>
   GITHUB_REPO=https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system
   ```

6. **Deploy:** Click "Deploy On-Demand" or "Deploy Spot"

### Step 2: Connect to Pod

```bash
# SSH into pod
ssh root@<pod-ip> -p <ssh-port>

# Or use Web Terminal
```

### Step 3: Run Quick Setup

```bash
# Download and run the quick setup script
cd /workspace
git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
cd adaptive-lab-system
bash runpod-quickstart.sh
```

This script will:
- ✅ Clone the repository
- ✅ Install system dependencies
- ✅ Install Python packages
- ✅ Verify GPU and CUDA
- ✅ Check HuggingFace authentication
- ✅ Test parameter extraction

### Step 4: Configure HuggingFace

```bash
# Login to HuggingFace
huggingface-cli login

# Enter your token when prompted
# Get token from: https://huggingface.co/settings/tokens
```

### Step 5: Start Backend

```bash
# Navigate to backend
cd /workspace/adaptive-lab-system/backend

# Start server
python main.py

# Server will run on http://0.0.0.0:8000
```

### Step 6: Test API

```bash
# In a new terminal/session

# Test health endpoint
curl http://localhost:8000/health

# Test upload
curl -X POST http://localhost:8000/api/upload \
  -F "file=@tests/fixtures/sample_procedures/DNA_Extraction.txt"

# Access API docs
# Open: http://<pod-ip>:8000/docs
```

### Step 7: Production Setup (Optional)

For production deployment, set up systemd service:

```bash
# Create service file
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

# Enable and start
systemctl daemon-reload
systemctl enable adaptive-lab-backend
systemctl start adaptive-lab-backend

# Check status
systemctl status adaptive-lab-backend
```

## 🧪 Testing Checklist

### Backend Tests
```bash
# Test parameter extraction
python test_parameter_extraction.py

# Test Llama integration (if you have access)
python scripts/test_llama_integration.py

# Verify setup
python scripts/verify_setup.py
```

### API Tests
```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST http://localhost:8000/api/upload \
  -F "file=@tests/fixtures/sample_procedures/DNA_Extraction.txt"

# Generate template
curl -X POST http://localhost:8000/api/generate-template \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Procedure",
    "steps": ["Add 50 µL buffer", "Incubate for 10 minutes at 25°C"],
    "user_id": "test_user"
  }'

# List templates
curl http://localhost:8000/api/templates
```

### Video Processing Tests
```bash
# Upload a test video to videos/

# Process with template
python scripts/test_video_with_tep.py \
  --template backend/templates/<template-id>.json \
  --video videos/test-video.mp4 \
  --output results/
```

## 📊 Monitoring

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check CUDA availability
python << EOF
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

### Check Service Status
```bash
# Backend service
systemctl status adaptive-lab-backend

# View logs
journalctl -u adaptive-lab-backend -f
```

### Monitor Disk Space
```bash
# Check disk usage
df -h

# Check directory sizes
du -sh /workspace/adaptive-lab-system/*
```

## 🔧 Troubleshooting Guide

### Issue: "Cannot access gated repo" (Llama)

**Status:** ⏳ Waiting for Llama 3 access

**Solution:**
1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
2. Click "Request Access"
3. Fill out the form and wait for approval
4. You'll receive an email when approved (1-2 days)

**Workaround:** System works perfectly with regex-only mode (no AI validation)

### Issue: "CUDA out of memory"

**Solution:**
- Upgrade to GPU with more VRAM (24GB minimum for Llama)
- Close other applications using GPU
- Clear CUDA cache: `torch.cuda.empty_cache()`

### Issue: Backend not accessible from outside

**Solution:**
1. Check RunPod port forwarding in pod settings
2. Use RunPod's proxy URL instead of direct IP
3. Ensure server binds to 0.0.0.0, not localhost

### Issue: Slow performance

**Solution:**
1. Check GPU is being used (not CPU): `nvidia-smi`
2. Ensure CUDA is available in Python
3. Consider upgrading GPU
4. Enable batch processing for multiple videos

## 📚 Documentation Reference

- **RUNPOD_SETUP.md** - Complete RunPod setup guide
- **HUGGINGFACE_ACCESS.md** - HuggingFace Llama 3 access
- **README.md** - General documentation
- **SETUP_GUIDE.md** - Local development setup
- **FRONTEND_SETUP.md** - Frontend-specific setup

## ✅ Final Checklist

Before going live, ensure:

- [ ] RunPod pod is running
- [ ] Repository cloned and updated
- [ ] All dependencies installed
- [ ] HuggingFace authenticated
- [ ] Llama 3 access approved (or regex-only mode accepted)
- [ ] Backend server running and accessible
- [ ] API endpoints tested
- [ ] GPU is detected and CUDA available
- [ ] Test template generated successfully
- [ ] Video processing tested (optional)
- [ ] Monitoring tools set up
- [ ] Systemd service configured (production)

## 🎯 Quick Command Reference

```bash
# Clone repository
git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git

# Run quick setup
bash runpod-quickstart.sh

# Start backend
cd backend && python main.py

# Test API
curl http://localhost:8000/health

# Check GPU
nvidia-smi

# View logs
journalctl -u adaptive-lab-backend -f

# Update code
git pull origin main
```

## 🔄 Updates and Maintenance

### Pulling Updates
```bash
cd /workspace/adaptive-lab-system
git pull origin main
pip install -r requirements.txt
systemctl restart adaptive-lab-backend
```

### Backup Templates
```bash
# Backup templates directory
tar -czf templates-backup-$(date +%Y%m%d).tar.gz backend/templates/

# Or sync to cloud storage
rclone copy backend/templates/ remote:backups/templates/
```

## 💰 Cost Optimization

1. **Use Spot Instances:** 3-5x cheaper (can be interrupted)
2. **Stop Pod When Idle:** Only pay for active time
3. **Start Small:** Test with smaller GPU, upgrade if needed
4. **Batch Processing:** Process multiple videos in one session
5. **Monitor Usage:** Track costs in RunPod dashboard

## 🎉 Success Criteria

Your deployment is successful when:

1. ✅ Backend responds to health checks
2. ✅ Templates can be generated from text/PDF
3. ✅ Parameters are extracted correctly
4. ✅ API docs are accessible
5. ✅ (Optional) Llama 3 validation works
6. ✅ (Optional) Videos can be processed

## 🚀 Next Steps After Deployment

1. **Test with Real Data:**
   - Upload your procedure PDFs
   - Generate templates
   - Review extracted parameters

2. **Process Videos:**
   - Upload lab procedure videos
   - Process with generated templates
   - Review results

3. **Integrate with Frontend:**
   - Build and deploy frontend
   - Connect to backend API
   - Share with team

4. **Scale:**
   - Add more templates
   - Process multiple videos
   - Optimize for your use case

## 📞 Support

- **GitHub Issues:** https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system/issues
- **RunPod Docs:** https://docs.runpod.io/
- **HuggingFace Support:** https://discuss.huggingface.co/

---

**Current Status:**
- ✅ GitHub: Ready
- ✅ Local Development: Working
- ⏳ HuggingFace: Awaiting Llama 3 access
- ⏳ RunPod: Ready to deploy

**Repository:** https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system
