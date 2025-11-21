"""
Convert YOLO/SAM/CLIP output to FrameData
Critical integration point between video processing and TEP
"""

from typing import Dict, List, Optional
import numpy as np

from ..tep.data_structures import FrameData


class FrameConverter:
    """
    Converts YOLO/SAM/CLIP detection output to TEP FrameData format

    IMPORTANT: This is the critical integration point.
    Adapt _convert_object() to match YOUR YOLO output format.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.object_counter = 0
        self.object_id_map = {}  # Track object IDs across frames

    def convert(
        self,
        yolo_result: Dict,
        frame_number: int,
        timestamp: float,
        imu_data: Optional[Dict] = None
    ) -> FrameData:
        """
        Convert YOLO output to FrameData format

        Args:
            yolo_result: Output from YOLO detection
            frame_number: Sequential frame number
            timestamp: Frame timestamp in seconds
            imu_data: Optional IMU sensor data

        Returns:
            FrameData object ready for TEP
        """
        objects = []

        # Get detected objects from YOLO result
        yolo_objects = yolo_result.get('objects', [])

        for yolo_obj in yolo_objects:
            converted = self._convert_object(yolo_obj, frame_number)
            if converted:
                objects.append(converted)

        return FrameData(
            timestamp=timestamp,
            frame_number=frame_number,
            objects=objects,
            imu_data=imu_data,
            workspace_id="default"
        )

    def _convert_object(self, yolo_obj: Dict, frame_number: int) -> Optional[Dict]:
        """
        Convert single YOLO object to TEP format

        YOUR YOLO OUTPUT FORMAT (from process_video.py):
        {
            'class': str,           # Class name (e.g., 'beaker', 'pipette')
            'confidence': float,    # Detection confidence
            'bbox': [x, y, w, h],   # Bounding box
            'mask': np.ndarray      # Optional segmentation mask
        }

        TEP REQUIRED FORMAT:
        {
            'id': str,              # Unique object ID
            'class': str,           # Object class
            'position': [x, y, z],  # 3D position (z from bbox size)
            'bbox': [x, y, w, h],   # Original bbox
            'confidence': float     # Detection confidence
        }
        """
        obj_class = yolo_obj.get('class', 'unknown')
        confidence = yolo_obj.get('confidence', 0.0)
        bbox = yolo_obj.get('bbox', [0, 0, 0, 0])

        # Generate consistent object ID based on class and position
        obj_id = self._get_object_id(obj_class, bbox, frame_number)

        # Convert bbox to 3D position
        # X, Y: center of bbox (normalized 0-1)
        # Z: estimated from bbox size (smaller = farther)
        position = self._bbox_to_position(bbox)

        return {
            'id': obj_id,
            'class': obj_class,
            'position': position,
            'bbox': bbox,
            'confidence': confidence,
            'is_active': False  # Will be set by window manager
        }

    def _get_object_id(self, obj_class: str, bbox: List, frame_number: int) -> str:
        """
        Generate consistent object ID for tracking across frames

        Uses spatial hashing to match objects across frames
        """
        if len(bbox) >= 4:
            # Create spatial hash
            center_x = (bbox[0] + bbox[2] / 2) / self.frame_width
            center_y = (bbox[1] + bbox[3] / 2) / self.frame_height
            spatial_hash = f"{obj_class}_{int(center_x * 10)}_{int(center_y * 10)}"

            if spatial_hash in self.object_id_map:
                return self.object_id_map[spatial_hash]
            else:
                self.object_counter += 1
                obj_id = f"{obj_class}_{self.object_counter}"
                self.object_id_map[spatial_hash] = obj_id
                return obj_id
        else:
            # Fallback: generate unique ID
            self.object_counter += 1
            return f"{obj_class}_{self.object_counter}"

    def _bbox_to_position(self, bbox: List) -> List[float]:
        """
        Convert 2D bounding box to estimated 3D position

        Uses bbox center for X, Y and size for Z (depth estimate)
        """
        if len(bbox) < 4:
            return [0.5, 0.5, 0.5]

        x, y, w, h = bbox

        # Normalize X, Y to 0-1 range
        center_x = (x + w / 2) / self.frame_width
        center_y = (y + h / 2) / self.frame_height

        # Estimate Z from bbox size (larger objects appear closer)
        # Normalize by frame diagonal
        diagonal = np.sqrt(self.frame_width**2 + self.frame_height**2)
        obj_size = np.sqrt(w**2 + h**2) / diagonal

        # Invert: larger size = smaller Z (closer)
        z = 1.0 - min(obj_size * 2, 0.9)

        return [center_x, center_y, z]

    def reset_tracking(self):
        """Reset object tracking for new video"""
        self.object_counter = 0
        self.object_id_map = {}


class BatchFrameConverter:
    """
    Converts batch of frames efficiently
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.converter = FrameConverter(frame_width, frame_height)

    def convert_batch(
        self,
        results: List[Dict],
        fps: float = 30.0
    ) -> List[FrameData]:
        """
        Convert list of YOLO results to list of FrameData

        Args:
            results: List of YOLO detection results
            fps: Video frame rate

        Returns:
            List of FrameData objects
        """
        frame_data_list = []

        for i, result in enumerate(results):
            timestamp = i / fps
            frame_data = self.converter.convert(
                yolo_result=result,
                frame_number=i,
                timestamp=timestamp
            )
            frame_data_list.append(frame_data)

        return frame_data_list

    def reset(self):
        """Reset for new video"""
        self.converter.reset_tracking()
