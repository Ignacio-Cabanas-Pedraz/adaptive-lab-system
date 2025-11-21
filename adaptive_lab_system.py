"""
Adaptive Lab System - Multi-mode object detection and tracking
Combines YOLOv9 (MIT license), SAM 2, and CLIP for flexible lab equipment detection
"""

import torch
import numpy as np
from enum import Enum
from yolo_wrapper import YOLO  # MIT-licensed YOLOv9 wrapper
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import clip
from PIL import Image
import time


class SystemMode(Enum):
    DISCOVERY = "discovery"      # Find everything
    TRACKING = "tracking"        # Track known objects
    VERIFICATION = "verification" # Quick checks


class AdaptiveLabSystem:
    def __init__(self, sam_checkpoint=None, sam_config=None, yolo_model='yolov9c.pt'):
        """
        Initialize all models once at startup

        Args:
            sam_checkpoint: Path to SAM 2 checkpoint (default: ./checkpoints/sam2.1_hiera_tiny.pt)
            sam_config: Path to SAM 2 config (default: ./configs/sam2.1/sam2.1_hiera_t.yaml)
            yolo_model: YOLOv9 model to use (default: yolov9c.pt - compact/balanced, MIT license)
        """

        print("=" * 60)
        print("Initializing Adaptive Lab System")
        print("=" * 60)

        # Set default paths
        if sam_checkpoint is None:
            sam_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        if sam_config is None:
            sam_config = "./configs/sam2.1/sam2.1_hiera_t.yaml"

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # YOLOv9 for fast detection (MIT license)
        print("\nLoading YOLOv9 model (MIT license)...")
        self.yolo = YOLO(yolo_model)
        print(f"✓ YOLOv9 {yolo_model} loaded (MIT license)")

        # SAM 2 for segmentation
        print("\nLoading SAM 2 models...")
        sam2_model = build_sam2(sam_config, sam_checkpoint, device=self.device)

        # SAM 2 - Automatic mask generation mode
        self.sam_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=16,  # Balanced speed/quality
            pred_iou_thresh=0.8,
            stability_score_thresh=0.9,
            min_mask_region_area=1000,
        )
        print("✓ SAM 2 Automatic mode loaded")

        # SAM 2 - Prompted segmentation mode
        sam2_model_prompted = build_sam2(sam_config, sam_checkpoint, device=self.device)
        self.sam_prompted = SAM2ImagePredictor(sam2_model_prompted)
        print("✓ SAM 2 Prompted mode loaded")

        # CLIP for classification
        print("\nLoading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        print("✓ CLIP ViT-B/32 loaded")

        # Lab equipment vocabulary
        self.lab_equipment_classes = [
            "a laboratory beaker",
            "a laboratory flask",
            "an erlenmeyer flask",
            "a test tube",
            "a microscope",
            "a pipette",
            "a bunsen burner",
            "a petri dish",
            "a laboratory scale",
            "human hands",
            "hands holding glassware",
            "safety goggles",
            "a graduated cylinder",
            "laboratory glassware",
        ]

        # Pre-encode text labels (do this once for speed)
        print("\nPre-encoding CLIP text features...")
        with torch.no_grad():
            text_tokens = clip.tokenize(self.lab_equipment_classes).to(self.device)
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print(f"✓ Encoded {len(self.lab_equipment_classes)} lab equipment classes")

        # System state
        self.mode = SystemMode.DISCOVERY
        self.known_objects = []
        self.frame_count = 0

        print("\n" + "=" * 60)
        print("✓ All models loaded successfully")
        print("=" * 60 + "\n")

    def set_mode(self, mode: SystemMode):
        """Change system operating mode"""
        self.mode = mode
        print(f"Mode changed to: {mode.value}")

    def process_frame(self, frame, context=None):
        """
        Main processing function - routes to appropriate mode

        Args:
            frame: numpy array (H, W, 3) RGB image
            context: dict with info like {"user_action": "exploring", "procedure_active": False}

        Returns:
            dict with detected objects, masks, and metadata
        """
        self.frame_count += 1
        start_time = time.time()

        # Auto-switch modes based on context
        if context:
            self.mode = self._determine_mode(context)

        # Route to appropriate processing pipeline
        if self.mode == SystemMode.DISCOVERY:
            result = self._discovery_mode(frame)
        elif self.mode == SystemMode.TRACKING:
            result = self._tracking_mode(frame)
        else:  # VERIFICATION
            result = self._verification_mode(frame)

        # Add actual processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result['actual_processing_time_ms'] = round(processing_time, 2)

        return result

    def _determine_mode(self, context):
        """Intelligently determine which mode to use"""

        # User explicitly exploring
        if context.get("user_action") == "exploring":
            return SystemMode.DISCOVERY

        # Following a procedure - need tracking
        if context.get("procedure_active"):
            return SystemMode.TRACKING

        # Quick safety check
        if context.get("emergency_check"):
            return SystemMode.VERIFICATION

        # Unknown object detected - switch to discovery
        if context.get("unknown_object_detected"):
            return SystemMode.DISCOVERY

        # First 3 frames - discover
        if self.frame_count < 3:
            return SystemMode.DISCOVERY

        # Default: tracking
        return SystemMode.TRACKING

    def _discovery_mode(self, frame):
        """
        DISCOVERY MODE: Find everything in the scene
        Uses: SAM 2 Auto + CLIP
        Speed: ~150-250ms
        """
        print(f"[DISCOVERY] Frame {self.frame_count}")

        # STEP 1: SAM 2 automatically finds all masks
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            all_masks = self.sam_auto.generate(frame)

        print(f"  Found {len(all_masks)} masks")

        # STEP 2: Classify each mask with CLIP
        detected_objects = []

        for idx, mask_data in enumerate(all_masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']
            area = mask_data['area']

            # Filter small masks
            if area < 500:
                continue

            # Crop region
            x, y, w, h = [int(v) for v in bbox]
            cropped = frame[y:y+h, x:x+w]

            # Classify with CLIP
            object_class, confidence = self._classify_with_clip(cropped)

            # Keep high-confidence lab equipment
            if confidence > 0.3:
                obj = {
                    'id': idx,
                    'class': object_class,
                    'confidence': float(confidence),
                    'mask': mask,
                    'bbox': bbox,
                    'area': area,
                    'center': self._get_center(mask)
                }
                detected_objects.append(obj)

        # Update known objects
        self.known_objects = detected_objects

        print(f"  Detected: {[obj['class'] for obj in detected_objects]}")

        return {
            'mode': 'discovery',
            'objects': detected_objects,
            'total_masks': len(all_masks),
            'frame_number': self.frame_count
        }

    def _tracking_mode(self, frame):
        """
        TRACKING MODE: Track known objects efficiently
        Uses: YOLOv9 + SAM 2 Prompted + CLIP
        Speed: ~50-100ms
        """
        print(f"[TRACKING] Frame {self.frame_count}")

        # STEP 1: YOLOv9 detects regions (~3ms)
        with torch.inference_mode():
            yolo_results = self.yolo(frame, verbose=False)

        detections = []
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                if conf > 0.5:
                    detections.append({
                        'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to x,y,w,h
                        'confidence': float(conf)
                    })

        print(f"  YOLOv9 found {len(detections)} regions")

        # STEP 2: SAM 2 segments each region (~10ms per region)
        tracked_objects = []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_prompted.set_image(frame)

            for idx, det in enumerate(detections):
                bbox = det['bbox']
                x, y, w, h = [int(v) for v in bbox]

                # Get precise mask from SAM 2
                masks, scores, _ = self.sam_prompted.predict(
                    box=np.array([x, y, x+w, y+h]),
                    multimask_output=False
                )

                mask = masks[0]

                # STEP 3: Classify with CLIP (~2ms)
                cropped = frame[y:y+h, x:x+w]
                object_class, clip_confidence = self._classify_with_clip(cropped)

                tracked_objects.append({
                    'id': idx,
                    'class': object_class,
                    'confidence': float(clip_confidence),
                    'mask': mask,
                    'bbox': bbox,
                    'center': self._get_center(mask)
                })

        print(f"  Tracking: {[obj['class'] for obj in tracked_objects]}")

        return {
            'mode': 'tracking',
            'objects': tracked_objects,
            'frame_number': self.frame_count
        }

    def _verification_mode(self, frame):
        """
        VERIFICATION MODE: Quick yes/no checks
        Uses: YOLOv9 + CLIP (no SAM 2)
        Speed: ~5-10ms
        """
        print(f"[VERIFICATION] Frame {self.frame_count}")

        # STEP 1: YOLOv9 detects (~3ms)
        with torch.inference_mode():
            yolo_results = self.yolo(frame, verbose=False)

        quick_checks = []

        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                if conf > 0.5:
                    # Quick CLIP classification on bounding box
                    x1, y1, x2, y2 = [int(v) for v in [x1, y1, x2, y2]]
                    cropped = frame[y1:y2, x1:x2]
                    object_class, clip_conf = self._classify_with_clip(cropped)

                    quick_checks.append({
                        'class': object_class,
                        'confidence': float(clip_conf),
                        'present': True
                    })

        print(f"  Quick check: {[obj['class'] for obj in quick_checks]}")

        return {
            'mode': 'verification',
            'objects': quick_checks,
            'frame_number': self.frame_count
        }

    def _classify_with_clip(self, cropped_region):
        """Classify a cropped image region using CLIP"""

        if cropped_region.size == 0 or cropped_region.shape[0] < 10 or cropped_region.shape[1] < 10:
            return "unknown", 0.0

        # Convert to PIL
        image = Image.fromarray(cropped_region)

        # Preprocess for CLIP
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compare with all lab equipment classes
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            confidence, best_idx = similarity[0].max(dim=0)

        # Clean up class name
        class_name = self.lab_equipment_classes[best_idx]
        class_name = class_name.replace("a ", "").replace("an ", "")

        return class_name, confidence.item()

    def _get_center(self, mask):
        """Get center point of a mask"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            return (0, 0)
        return (int(np.mean(x_indices)), int(np.mean(y_indices)))
