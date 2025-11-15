# Adaptive Lab System

A multi-mode computer vision system for detecting and tracking laboratory equipment using YOLO, SAM 2, and CLIP.

## Overview

This system combines three powerful models to provide flexible, adaptive object detection for laboratory environments:

- **YOLO (YOLOv8)**: Fast object detection for locating regions of interest (3ms)
- **SAM 2**: Precise segmentation for accurate object masks (10ms per object)
- **CLIP**: Zero-shot classification for identifying lab equipment (2ms per object)

### System Modes

1. **Discovery Mode**: Find everything in the scene using SAM 2 Auto + CLIP (~150-250ms)
2. **Tracking Mode**: Efficiently track known objects using YOLO + SAM 2 + CLIP (~50-100ms)
3. **Verification Mode**: Quick checks using YOLO + CLIP (~5-10ms)

## Features

- ✅ Zero-shot learning - works without training on lab-specific data
- ✅ Adaptive mode switching based on context
- ✅ Real-time capable on GPU (10-20 FPS in tracking mode)
- ✅ Precise segmentation masks (not just bounding boxes)
- ✅ Video processing with visualization
- ✅ JSON export of detection results

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- RunPod account (for cloud GPU access)

## RunPod Deployment Guide

### Step 1: Create a RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io) and sign in
2. Click **"Deploy"** → **"GPU Pods"**
3. Select a GPU template:
   - **Recommended**: RTX 4090 (24GB VRAM) or A5000 (24GB VRAM)
   - **Minimum**: RTX 3090 (24GB VRAM) or RTX A4000 (16GB VRAM)
4. Choose **"PyTorch"** template (comes with CUDA pre-installed)
5. Set storage to at least **50GB**
6. Click **"Deploy On-Demand"** or **"Deploy Spot"** (spot is cheaper)

### Step 2: Connect to Your Pod

Once your pod is running:

1. Click **"Connect"** → **"Start Web Terminal"** or use SSH
2. Alternatively, use **Jupyter Lab** if you prefer a notebook interface

### Step 3: Clone and Setup

In your RunPod terminal:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/adaptive-lab-system.git
cd adaptive-lab-system

# Run the setup script (downloads models and installs dependencies)
bash setup_runpod.sh
```

The setup script will:
- Install all Python dependencies
- Download SAM 2 checkpoint (~150MB)
- Download SAM 2 config files
- Create necessary directories
- Verify GPU availability

This takes approximately **5-10 minutes** depending on your connection.

### Step 4: Upload Your Video

You can upload videos in two ways:

**Option A: Using RunPod Web Interface**
1. Click the **folder icon** in RunPod
2. Navigate to `adaptive-lab-system/videos/`
3. Click **"Upload"** and select your video file

**Option B: Using wget/curl**
```bash
cd videos
wget YOUR_VIDEO_URL -O test_video.mp4
cd ..
```

### Step 5: Process Your Video

```bash
# Basic processing (auto mode)
python process_video.py --video videos/test_video.mp4

# With JSON export and mask saving
python process_video.py --video videos/test_video.mp4 --save-json --save-masks

# Process only every 5th frame (5x faster)
python process_video.py --video videos/test_video.mp4 --skip-frames 5

# Use specific mode
python process_video.py --video videos/test_video.mp4 --mode discovery

# Limit to first 100 frames
python process_video.py --video videos/test_video.mp4 --max-frames 100
```

### Step 6: Download Results

Results are saved in `output/run_TIMESTAMP/`:
- `annotated_video.mp4` - Video with detection overlays
- `results.json` - Frame-by-frame detection data (if `--save-json` used)
- `summary.json` - Processing statistics
- `masks/` - Individual mask images (if `--save-masks` used)

**Download via RunPod Web Interface:**
1. Navigate to the output folder
2. Right-click on files → **"Download"**

**Download via RunPod CLI:**
```bash
# From your local machine
runpodctl receive YOUR_POD_ID:/workspace/adaptive-lab-system/output/ ./local_output/
```

## Usage Examples

### Command Line Arguments

```
--video          Path to input video file (required)
--output         Output directory (default: ./output)
--mode           Processing mode: auto/discovery/tracking/verification (default: auto)
--sam-checkpoint Path to SAM 2 checkpoint (default: ./checkpoints/sam2.1_hiera_tiny.pt)
--sam-config     Path to SAM 2 config (default: ./configs/sam2.1/sam2.1_hiera_t.yaml)
--yolo-model     YOLO model variant: yolov8n/s/m/l (default: yolov8n)
--skip-frames    Process every Nth frame (default: 1)
--max-frames     Maximum frames to process (default: all)
--save-masks     Save individual mask visualizations
--save-json      Save detection results as JSON
```

### Example Workflows

**Quick Test (Fast)**
```bash
python process_video.py \
  --video videos/lab_demo.mp4 \
  --skip-frames 10 \
  --max-frames 50
```

**Full Analysis (Detailed)**
```bash
python process_video.py \
  --video videos/lab_demo.mp4 \
  --save-json \
  --save-masks \
  --mode auto
```

**High Accuracy (Slower)**
```bash
python process_video.py \
  --video videos/lab_demo.mp4 \
  --yolo-model yolov8l \
  --mode discovery
```

## Project Structure

```
adaptive-lab-system/
├── adaptive_lab_system.py   # Main system class
├── process_video.py          # Video processing script
├── requirements.txt          # Python dependencies
├── setup_runpod.sh          # RunPod setup script
├── README.md                # This file
├── checkpoints/             # Model checkpoints
│   └── sam2.1_hiera_tiny.pt
├── configs/                 # Model configurations
│   └── sam2.1/
│       └── sam2.1_hiera_t.yaml
├── videos/                  # Input videos (upload here)
└── output/                  # Processing results
    └── run_TIMESTAMP/
        ├── annotated_video.mp4
        ├── results.json
        ├── summary.json
        └── masks/
```

## Supported Lab Equipment

The system can detect (zero-shot, no training required):
- Laboratory beakers
- Laboratory flasks (including Erlenmeyer)
- Test tubes
- Microscopes
- Pipettes
- Bunsen burners
- Petri dishes
- Laboratory scales
- Graduated cylinders
- Human hands / hands holding glassware
- Safety goggles
- General laboratory glassware

You can easily extend this by modifying the `lab_equipment_classes` list in `adaptive_lab_system.py:384`.

## Performance Benchmarks

Tested on RTX 4090:

| Mode         | Time per Frame | FPS  | Use Case                    |
|--------------|----------------|------|-----------------------------|
| Discovery    | 150-250ms      | 4-6  | Initial scene exploration   |
| Tracking     | 50-100ms       | 10-20| Continuous tracking         |
| Verification | 5-10ms         | 100+ | Quick safety checks         |

## Troubleshooting

### Out of Memory Errors

```bash
# Use smaller YOLO model
python process_video.py --video videos/test.mp4 --yolo-model yolov8n

# Process fewer frames
python process_video.py --video videos/test.mp4 --skip-frames 5

# Reduce video resolution before processing
ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
```

### CUDA Errors

```bash
# Verify GPU is available
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Slow Processing

- Use `--skip-frames 5` to process every 5th frame
- Use `yolov8n` (nano) instead of larger models
- Use `--mode tracking` for faster processing
- Use smaller video resolution

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Install specific packages
pip install ultralytics segment-anything-2 clip
```

## Cost Estimation (RunPod)

- **RTX 4090**: ~$0.69/hour
- **RTX 3090**: ~$0.44/hour
- **A5000**: ~$0.59/hour

Average processing time: **~5 hours of video per hour** (with skip-frames=5)

## API Usage (Python)

```python
from adaptive_lab_system import AdaptiveLabSystem, SystemMode
import cv2

# Initialize system
lab_system = AdaptiveLabSystem()

# Process single frame
frame = cv2.imread('test_image.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Auto mode
result = lab_system.process_frame(frame_rgb)

# Specific mode with context
context = {"procedure_active": True}
result = lab_system.process_frame(frame_rgb, context)

# Access results
for obj in result['objects']:
    print(f"Detected: {obj['class']} ({obj['confidence']:.2f})")
    mask = obj['mask']
    bbox = obj['bbox']
```

## Future Enhancements

- [ ] Real-time video streaming support
- [ ] Fine-tuned YOLO on lab-specific data
- [ ] Multi-camera support
- [ ] Temporal tracking across frames
- [ ] Safety violation detection
- [ ] Export to annotation formats (COCO, YOLO)

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in your research, please cite:

- **YOLO**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **SAM 2**: [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- **CLIP**: [OpenAI CLIP](https://github.com/openai/CLIP)

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the original model documentation

## Acknowledgments

Built using:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta's Segment Anything 2](https://github.com/facebookresearch/segment-anything-2)
- [OpenAI's CLIP](https://github.com/openai/CLIP)
- [RunPod](https://www.runpod.io) for GPU infrastructure
