# Troubleshooting Guide

Common issues and solutions for the Adaptive Lab System on RunPod.

## 1. "No space left on device" Error

### Symptoms
```
OSError: [Errno 28] No space left on device
```

### Solution A: Run the cleanup script
```bash
bash cleanup_disk.sh
```

### Solution B: Manual cleanup
```bash
# Check current disk usage
df -h

# Clean pip cache (can free several GB)
pip cache purge

# Clean apt cache
apt-get clean
apt-get autoremove -y

# Remove temp files
rm -rf /tmp/*
rm -rf /var/tmp/*

# Check space again
df -h
```

### Solution C: Reinstall with no cache
```bash
# Remove and reinstall packages without cache
pip uninstall -y torch torchvision
pip install --no-cache-dir -r requirements.txt
```

### Solution D: Increase RunPod storage (Recommended)
1. Stop your pod
2. Click "Edit" on your pod
3. Increase "Container Disk" to **50-100GB**
4. Restart the pod
5. Run `bash setup_runpod.sh` again

---

## 2. CUDA Out of Memory Error

### Symptoms
```
RuntimeError: CUDA out of memory
```

### Solutions

**Use smaller models:**
```bash
# Use nano YOLO instead of larger models
python process_video.py --video videos/test.mp4 --yolo-model yolov8n
```

**Process fewer frames:**
```bash
# Process every 10th frame instead of every frame
python process_video.py --video videos/test.mp4 --skip-frames 10
```

**Use verification mode (fastest, least memory):**
```bash
python process_video.py --video videos/test.mp4 --mode verification
```

**Reduce video resolution:**
```bash
# Downscale video before processing
ffmpeg -i input.mp4 -vf scale=1280:720 output_720p.mp4
```

**Use a larger GPU:**
- Recommended: RTX 4090 (24GB VRAM)
- Minimum: RTX 3090 (24GB VRAM)

---

## 3. Import Errors

### Symptoms
```
ModuleNotFoundError: No module named 'sam2'
ImportError: cannot import name 'YOLO'
```

### Solution
```bash
# Reinstall dependencies
pip install --no-cache-dir -r requirements.txt

# Or install individually
pip install --no-cache-dir ultralytics
pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git
pip install --no-cache-dir git+https://github.com/openai/CLIP.git
```

### Verify installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('YOLO: OK')"
python -c "import sam2; print('SAM 2: OK')"
python -c "import clip; print('CLIP: OK')"
```

---

## 4. SAM 2 Checkpoint Not Found

### Symptoms
```
FileNotFoundError: checkpoints/sam2.1_hiera_tiny.pt not found
```

### Solution
```bash
# Download SAM 2 checkpoint manually
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
cd ..
```

### Download config file
```bash
mkdir -p configs/sam2.1
cd configs/sam2.1
wget https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
cd ../..
```

---

## 5. Video Processing is Too Slow

### Solutions

**Skip frames:**
```bash
# Process every 5th frame (5x faster)
python process_video.py --video videos/test.mp4 --skip-frames 5
```

**Use tracking or verification mode:**
```bash
# Tracking mode is faster than discovery
python process_video.py --video videos/test.mp4 --mode tracking

# Verification mode is fastest
python process_video.py --video videos/test.mp4 --mode verification
```

**Use smaller YOLO model:**
```bash
python process_video.py --video videos/test.mp4 --yolo-model yolov8n
```

**Limit frames processed:**
```bash
# Only process first 100 frames
python process_video.py --video videos/test.mp4 --max-frames 100
```

---

## 6. CUDA/GPU Not Detected

### Symptoms
```
CUDA Available: False
```

### Solutions

**Verify GPU:**
```bash
nvidia-smi
```

**Check PyTorch CUDA:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Reinstall PyTorch with correct CUDA version:**
```bash
# For CUDA 11.8
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Use RunPod PyTorch template:**
- When creating pod, select "PyTorch" or "CUDA" base image
- These come with GPU drivers pre-installed

---

## 7. Video File Not Found

### Symptoms
```
ValueError: Could not open video file: videos/test.mp4
```

### Solutions

**Verify file exists:**
```bash
ls -lh videos/
```

**Check file path:**
```bash
# Use absolute path if needed
python process_video.py --video /workspace/adaptive-lab-system/videos/test.mp4
```

**Upload video to RunPod:**
- Use RunPod web interface file browser
- Upload to `adaptive-lab-system/videos/` directory
- Or use `wget` to download from URL:
  ```bash
  cd videos
  wget YOUR_VIDEO_URL -O test.mp4
  cd ..
  ```

---

## 8. Output Video is Corrupted

### Symptoms
- Output video won't play
- Black frames
- Error when opening MP4

### Solutions

**Install ffmpeg:**
```bash
apt-get install -y ffmpeg
```

**Convert video format:**
```bash
# Convert to MP4 with H.264
ffmpeg -i output/run_*/annotated_video.mp4 -vcodec libx264 -acodec aac output_fixed.mp4
```

**Check OpenCV installation:**
```bash
pip install --no-cache-dir opencv-python opencv-contrib-python --force-reinstall
```

---

## 9. Permission Denied Errors

### Symptoms
```
PermissionError: [Errno 13] Permission denied
```

### Solutions

**Make scripts executable:**
```bash
chmod +x setup_runpod.sh
chmod +x cleanup_disk.sh
```

**Run as root (RunPod default):**
```bash
sudo bash setup_runpod.sh
```

**Check file permissions:**
```bash
ls -la
```

---

## 10. Git Clone/Pull Errors

### Symptoms
```
fatal: could not read Username
Permission denied (publickey)
```

### Solutions

**Use HTTPS instead of SSH:**
```bash
git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
```

**Update existing repo:**
```bash
cd adaptive-lab-system
git remote set-url origin https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
git pull
```

---

## Getting More Help

### Check System Info
```bash
# Python version
python --version

# PyTorch info
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Disk space
df -h

# GPU info
nvidia-smi

# Memory info
free -h
```

### Enable Verbose Output
```bash
# Run with verbose mode to see detailed errors
python process_video.py --video videos/test.mp4 --save-json 2>&1 | tee debug.log
```

### Report Issues
If you encounter issues not covered here:
1. Check the error message carefully
2. Search existing GitHub issues
3. Open a new issue with:
   - Error message
   - Command you ran
   - System info (`nvidia-smi`, `python --version`)
   - Disk space (`df -h`)

---

## Performance Optimization Tips

### Best Practices

1. **Use appropriate mode:**
   - Discovery: Initial scene exploration (slow, thorough)
   - Tracking: Continuous monitoring (fast, real-time)
   - Verification: Quick checks (very fast)

2. **Optimize for your use case:**
   - High accuracy: `--yolo-model yolov8l --mode discovery`
   - Balanced: `--yolo-model yolov8n --mode tracking`
   - Speed: `--mode verification --skip-frames 5`

3. **Monitor resources:**
   ```bash
   # Watch GPU usage in real-time
   watch -n 1 nvidia-smi

   # Monitor disk space
   watch -n 5 df -h
   ```

4. **Clean up regularly:**
   ```bash
   # After each run
   bash cleanup_disk.sh

   # Remove old outputs
   rm -rf output/run_*
   ```

---

## Quick Reference

### Essential Commands
```bash
# Setup
bash setup_runpod.sh

# Clean disk
bash cleanup_disk.sh

# Basic processing
python process_video.py --video videos/test.mp4

# Fast processing
python process_video.py --video videos/test.mp4 --skip-frames 5 --mode tracking

# Full analysis
python process_video.py --video videos/test.mp4 --save-json --save-masks

# Check status
df -h          # Disk space
nvidia-smi     # GPU status
git status     # Repository status
```
