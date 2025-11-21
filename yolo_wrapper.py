"""
YOLOv9 Wrapper for WongKinYiu MIT-licensed implementation
Provides compatibility layer with Ultralytics-style API
"""

import torch
import numpy as np
from pathlib import Path


class YOLOv9Wrapper:
    """
    Wrapper for WongKinYiu's YOLOv9 implementation (MIT license)
    Provides similar interface to Ultralytics YOLO for compatibility
    """

    def __init__(self, model_name='v9-c', device=None, repo='WongKinYiu/yolov9'):
        """
        Initialize YOLOv9 model

        Args:
            model_name: Model variant (v9-t, v9-s, v9-m, v9-c, v9-e)
            device: 'cuda' or 'cpu' (auto-detected if None)
            repo: GitHub repository for torch.hub
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLOv9 ({model_name}) from {repo}...")

        # Map model names to expected format
        # yolov9t.pt -> v9-t, yolov9c.pt -> v9-c, etc.
        if model_name.endswith('.pt'):
            # Extract variant (t, s, m, c, e)
            variant = model_name.replace('yolov9', '').replace('.pt', '')
            hub_model = f'yolov9{variant}'
        else:
            hub_model = model_name

        try:
            # Load model using torch.hub
            # This will download the model if not cached
            self.model = torch.hub.load(
                repo,
                'custom',
                path=hub_model,
                trust_repo=True,
                force_reload=False
            )

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            print(f"✓ YOLOv9 model loaded on {self.device}")

        except Exception as e:
            print(f"Error loading from torch.hub: {e}")
            print("Attempting to load local weights...")

            # Fallback: Try loading local weights
            self._load_local_weights(hub_model)

    def _load_local_weights(self, model_name):
        """
        Fallback: Load local weights if torch.hub fails
        """
        # Try common locations
        weight_paths = [
            f"./{model_name}.pt",
            f"./weights/{model_name}.pt",
            f"./checkpoints/{model_name}.pt",
        ]

        for weight_path in weight_paths:
            if Path(weight_path).exists():
                print(f"Loading local weights from {weight_path}")
                self.model = torch.hub.load(
                    'WongKinYiu/yolov9',
                    'custom',
                    path=weight_path,
                    trust_repo=True
                )
                self.model.to(self.device)
                self.model.eval()
                return

        raise FileNotFoundError(
            f"Could not load model {model_name}. "
            f"Please download weights from: "
            f"https://github.com/WongKinYiu/yolov9/releases"
        )

    def __call__(self, image, verbose=True, conf=0.25, iou=0.45, img_size=640):
        """
        Run inference on image

        Args:
            image: numpy array (H, W, 3) in RGB format
            verbose: Print results
            conf: Confidence threshold
            iou: IOU threshold for NMS
            img_size: Input image size

        Returns:
            List of detection results in Ultralytics-compatible format
        """
        # Set confidence and IOU thresholds
        self.model.conf = conf
        self.model.iou = iou

        # Run inference
        with torch.no_grad():
            results = self.model(image, size=img_size)

        # Wrap results in compatible format
        return [YOLOv9Result(results, image.shape)]

    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model.to(device)
        return self


class YOLOv9Result:
    """
    Wrapper for YOLOv9 results to provide Ultralytics-compatible interface
    """

    def __init__(self, results, image_shape):
        """
        Initialize result wrapper

        Args:
            results: YOLOv9 inference results
            image_shape: Original image shape (H, W, C)
        """
        self.results = results
        self.image_shape = image_shape
        self._boxes = None

    @property
    def boxes(self):
        """
        Get boxes in Ultralytics-compatible format

        Returns:
            YOLOv9Boxes object with xyxy, conf properties
        """
        if self._boxes is None:
            self._boxes = YOLOv9Boxes(self.results)
        return self._boxes


class YOLOv9Boxes:
    """
    Wrapper for detection boxes to provide Ultralytics-compatible interface
    """

    def __init__(self, results):
        """
        Initialize boxes wrapper

        Args:
            results: YOLOv9 results object
        """
        self.results = results

        # Extract detections from results
        # YOLOv9 results have .xyxy[0] containing [x1, y1, x2, y2, conf, class]
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            self.detections = results.xyxy[0].cpu()  # Move to CPU
        else:
            self.detections = torch.empty((0, 6))

    @property
    def xyxy(self):
        """Get bounding boxes in xyxy format"""
        if len(self.detections) == 0:
            return torch.empty((0, 4))
        return self.detections[:, :4]

    @property
    def conf(self):
        """Get confidence scores"""
        if len(self.detections) == 0:
            return torch.empty(0)
        return self.detections[:, 4]

    @property
    def cls(self):
        """Get class IDs"""
        if len(self.detections) == 0:
            return torch.empty(0)
        return self.detections[:, 5]

    def __len__(self):
        """Number of detections"""
        return len(self.detections)

    def __iter__(self):
        """Iterate over boxes"""
        for i in range(len(self.detections)):
            yield YOLOv9Box(self.detections[i])


class YOLOv9Box:
    """
    Single detection box wrapper
    """

    def __init__(self, detection):
        """
        Initialize box

        Args:
            detection: Single detection tensor [x1, y1, x2, y2, conf, class]
        """
        self.detection = detection

    @property
    def xyxy(self):
        """Bounding box coordinates"""
        return self.detection[:4].unsqueeze(0)

    @property
    def conf(self):
        """Confidence score"""
        return self.detection[4].unsqueeze(0)

    @property
    def cls(self):
        """Class ID"""
        return self.detection[5].unsqueeze(0)


def YOLO(model_path='yolov9c.pt'):
    """
    Factory function for backward compatibility

    Args:
        model_path: Path to model or model name

    Returns:
        YOLOv9Wrapper instance
    """
    return YOLOv9Wrapper(model_name=model_path)
