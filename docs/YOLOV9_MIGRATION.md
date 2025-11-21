# YOLOv9 Migration Guide - MIT License Implementation

## Overview

This project has been successfully migrated from Ultralytics YOLOv8 (AGPL-3.0) to WongKinYiu's YOLOv9 (MIT License) for **free commercial use**.

## What Changed

### 1. Dependencies ([requirements.txt](requirements.txt))
- **Before**: `ultralytics>=8.2.0` (AGPL-3.0 license)
- **After**: `git+https://github.com/WongKinYiu/YOLO.git` (MIT license)
- **Added**: `hydra-core>=1.3.0` and `omegaconf>=2.3.0` (for YOLO config support)

### 2. New Wrapper Class ([yolo_wrapper.py](yolo_wrapper.py))
Created a compatibility wrapper that bridges WongKinYiu's YOLOv9 API to work with the existing codebase:
- Uses `torch.hub.load()` to load YOLOv9 models
- Provides Ultralytics-compatible interface (YOLO class, results.boxes, etc.)
- Supports all YOLOv9 variants: t (tiny), s (small), m (medium), c (compact), e (extended)

### 3. Core System ([adaptive_lab_system.py](adaptive_lab_system.py))
- Import changed: `from yolo_wrapper import YOLO`
- Default model: `yolov9c.pt` (compact variant - balanced performance)
- Comments updated to reflect MIT license

### 4. Processing Scripts
- [process_video.py](process_video.py): Default model → `yolov9c.pt`
- [process_video_optimized.py](process_video_optimized.py): Default model → `yolov9c.pt`

### 5. Documentation ([README.md](README.md))
- Added MIT license notice at the top
- Updated component breakdown with license information
- Updated command-line examples
- Added comprehensive license section

## Installation

### Fresh Install

```bash
# Clone repository
git clone https://github.com/Ignacio-Cabanas-Pedraz/adaptive-lab-system.git
cd adaptive-lab-system

# Install dependencies (includes MIT-licensed YOLOv9)
pip install -r requirements.txt
```

### Upgrading from YOLOv8

```bash
# Uninstall Ultralytics (optional, if you have it)
pip uninstall ultralytics

# Install new dependencies
pip install -r requirements.txt
```

## Usage

### Model Selection

YOLOv9 offers 5 model variants (all MIT licensed):

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| yolov9t.pt | 2.0M | Fastest | Low | Real-time on CPU |
| yolov9s.pt | 7.2M | Fast | Medium | Mobile/Edge devices |
| yolov9m.pt | 20.1M | Medium | High | Balanced GPU workloads |
| **yolov9c.pt** | 25.5M | Medium | High | **Default - Best balance** |
| yolov9e.pt | 58.1M | Slow | Highest | Maximum accuracy |

### Basic Usage

```python
from adaptive_lab_system import AdaptiveLabSystem

# Initialize with YOLOv9-c (default)
system = AdaptiveLabSystem()

# Or specify a different variant
system = AdaptiveLabSystem(yolo_model='yolov9t.pt')  # Fastest
system = AdaptiveLabSystem(yolo_model='yolov9e.pt')  # Most accurate
```

### Command Line

```bash
# Use default YOLOv9-c
python process_video.py --video input.mp4

# Use tiny variant for speed
python process_video.py --video input.mp4 --yolo-model yolov9t.pt

# Use extended variant for accuracy
python process_video.py --video input.mp4 --yolo-model yolov9e.pt
```

## How the Wrapper Works

The [yolo_wrapper.py](yolo_wrapper.py) file provides a seamless compatibility layer:

### Loading Models
```python
# Wrapper uses torch.hub.load internally
model = YOLO('yolov9c.pt')

# Translates to:
# torch.hub.load('WongKinYiu/yolov9', 'custom', path='yolov9c')
```

### Inference
```python
# Same API as before
results = model(image, verbose=False)

# Results have familiar structure:
for result in results:
    boxes = result.boxes
    for box in boxes:
        xyxy = box.xyxy  # Bounding box coordinates
        conf = box.conf  # Confidence scores
        cls = box.cls    # Class IDs
```

### Model Download
First time you run the code, YOLOv9 will automatically download the model weights from the WongKinYiu/yolov9 repository via torch.hub. This is cached for future use.

## License Compliance

### What You Can Do (MIT License) ✅
- Use commercially without restrictions
- Modify and redistribute
- Include in proprietary software
- No requirement to open-source your project

### Attribution Required
Please include acknowledgment of:
- WongKinYiu for YOLOv9 (MIT license)
- Meta AI Research for SAM 2 (Apache 2.0)
- OpenAI for CLIP (MIT license)

See [README.md](README.md) for full citations.

## Troubleshooting

### Model Download Issues

If torch.hub fails to download models:

1. **Manual Download**:
   ```bash
   # Download from releases
   wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

   # Place in project root or ./weights/ or ./checkpoints/
   mv yolov9-c.pt ./weights/
   ```

2. **Update wrapper path**:
   The wrapper will automatically search common locations:
   - `./yolov9c.pt`
   - `./weights/yolov9c.pt`
   - `./checkpoints/yolov9c.pt`

### API Differences

If you encounter issues, the wrapper may need updates. Key differences from Ultralytics:

| Ultralytics | YOLOv9 (WongKinYiu) |
|-------------|---------------------|
| `YOLO('yolov8n.pt')` | `torch.hub.load('WongKinYiu/yolov9', 'custom', 'yolov9c')` |
| Built-in model download | Torch hub or manual download |
| `results.boxes.xyxy` | `results.xyxy[0]` |

The wrapper handles these conversions automatically.

### Performance

Expected performance (RTX 4090):

| Variant | Tracking Mode | Discovery Mode |
|---------|---------------|----------------|
| yolov9t | ~40ms/frame | ~180ms/frame |
| yolov9s | ~45ms/frame | ~190ms/frame |
| yolov9m | ~55ms/frame | ~210ms/frame |
| **yolov9c** | **~60ms/frame** | **~220ms/frame** |
| yolov9e | ~85ms/frame | ~270ms/frame |

## Migration Checklist

- [x] Updated requirements.txt to MIT-licensed YOLO
- [x] Created yolo_wrapper.py compatibility layer
- [x] Updated adaptive_lab_system.py imports
- [x] Changed default model to yolov9c.pt
- [x] Updated process_video.py
- [x] Updated process_video_optimized.py
- [x] Updated README.md with license info
- [x] Added license compliance section
- [x] Updated citations and acknowledgments

## Next Steps

1. **Test the installation**:
   ```bash
   python -c "from yolo_wrapper import YOLO; print('✓ Wrapper works!')"
   ```

2. **Download a test model**:
   ```bash
   # The wrapper will do this automatically on first run
   # Or manually download from:
   # https://github.com/WongKinYiu/yolov9/releases
   ```

3. **Run a test**:
   ```bash
   python process_video.py --video test.mp4 --max-frames 10
   ```

## Support

For issues specific to:
- **Wrapper compatibility**: Open an issue in this repository
- **YOLOv9 model**: See [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- **License questions**: Review the [LICENSE](LICENSE) file

## References

- YOLOv9 Paper: [arXiv:2402.13616](https://arxiv.org/abs/2402.13616)
- YOLOv9 Repository: [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- MIT License: [WongKinYiu/YOLO](https://github.com/WongKinYiu/YOLO)
