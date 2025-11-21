"""
Temporal Window Manager
Manages temporal windows for action detection with boundary detection
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .data_structures import FrameData, TemporalWindow, BoundaryType


class TemporalWindowManager:
    """
    Manages temporal windows for action detection
    Enhanced with uncertainty tracking and smart boundary detection
    """

    def __init__(self, window_size: float = 2.0, fps: int = 30):
        self.window_size = window_size
        self.fps = fps
        self.max_frames = int(window_size * fps)

        self.frame_buffer = deque(maxlen=self.max_frames)
        self.current_step = None

        # Motion tracking for boundary detection
        self.motion_history = deque(maxlen=30)  # 1 second of motion
        self.stability_threshold = 0.1

    def update(self, frame_data: FrameData, current_step: Dict) -> Optional[TemporalWindow]:
        """
        Add frame to buffer and check for action completion

        Returns TemporalWindow if action boundary detected
        """
        self.frame_buffer.append(frame_data)
        self.current_step = current_step

        # Calculate motion metric
        motion = self._calculate_motion_metric(frame_data)
        self.motion_history.append(motion)

        # Expected action guides boundary detection
        expected_action = current_step.get('expected_action')
        expected_objects = current_step.get('expected_objects', [])

        # Focus on expected objects
        active_objects = self._detect_active_objects(
            frame_data,
            expected_objects
        )

        # Check for action boundary
        boundary_detected, boundary_type = self._detect_action_boundary(
            active_objects,
            expected_action
        )

        if boundary_detected:
            return self._extract_window_with_uncertainty(boundary_type)

        return None

    def _calculate_motion_metric(self, frame_data: FrameData) -> float:
        """
        Calculate overall motion in frame
        Uses IMU + object position changes
        """
        # IMU-based motion
        imu_motion = 0.0
        if frame_data.imu_data:
            accel = frame_data.imu_data.get('accelerometer', [0, 0, 0])
            imu_motion = np.linalg.norm(accel)

        # Visual motion (if we have previous frame)
        visual_motion = 0.0
        if len(self.frame_buffer) > 0:
            prev_frame = self.frame_buffer[-1]
            visual_motion = self._calculate_visual_motion(
                prev_frame.objects,
                frame_data.objects
            )

        # Combine
        return 0.3 * imu_motion + 0.7 * visual_motion

    def _calculate_visual_motion(
        self,
        prev_objects: List[Dict],
        curr_objects: List[Dict]
    ) -> float:
        """
        Calculate motion between frames based on object positions
        """
        if not prev_objects or not curr_objects:
            return 0.0

        # Match objects by ID
        motion_sum = 0.0
        matched_count = 0

        prev_dict = {obj['id']: obj for obj in prev_objects}
        curr_dict = {obj['id']: obj for obj in curr_objects}

        for obj_id in prev_dict.keys() & curr_dict.keys():
            prev_pos = np.array(prev_dict[obj_id]['position'])
            curr_pos = np.array(curr_dict[obj_id]['position'])
            motion_sum += np.linalg.norm(curr_pos - prev_pos)
            matched_count += 1

        return motion_sum / max(matched_count, 1)

    def _detect_active_objects(
        self,
        frame_data: FrameData,
        expected_objects: List[str]
    ) -> Set[str]:
        """
        Identify objects currently being manipulated
        Prioritizes expected objects from template
        """
        active = set()

        # Check motion of each object
        if len(self.frame_buffer) >= 2:
            prev_frame = self.frame_buffer[-2]

            for obj in frame_data.objects:
                obj_id = obj['id']

                # Priority to expected objects
                is_expected = obj_id in expected_objects

                # Check if object moved
                prev_obj = next(
                    (o for o in prev_frame.objects if o['id'] == obj_id),
                    None
                )

                if prev_obj:
                    movement = np.linalg.norm(
                        np.array(obj['position']) - np.array(prev_obj['position'])
                    )

                    threshold = 0.05 if is_expected else 0.1
                    if movement > threshold:
                        active.add(obj_id)

        return active

    def _detect_action_boundary(
        self,
        active_objects: Set[str],
        expected_action: Optional[str]
    ) -> Tuple[bool, BoundaryType]:
        """
        Detect if an action has completed
        Returns (boundary_detected, boundary_type)
        """
        if len(self.motion_history) < 10:
            return False, BoundaryType.SHARP

        recent_motion = list(self.motion_history)[-10:]
        avg_motion = np.mean(recent_motion)
        motion_stable = avg_motion < self.stability_threshold

        if expected_action == 'transfer':
            # Transfer: sharp boundaries, look for approach-withdraw
            detected = self._detect_approach_withdraw_pattern(active_objects)
            return detected, BoundaryType.SHARP

        elif expected_action == 'mix':
            # Mix: gradual boundaries, look for oscillation stop
            detected = self._detect_oscillation_stop()
            return detected, BoundaryType.GRADUAL

        elif expected_action in ['heat', 'cool']:
            # Heat/cool: continuous, look for placement stability
            detected = self._detect_placement_stability(active_objects)
            return detected, BoundaryType.CONTINUOUS

        elif expected_action == 'wait':
            # Wait: continuous, look for sustained inactivity
            detected = motion_stable and len(self.frame_buffer) >= 30
            return detected, BoundaryType.CONTINUOUS

        # Default: motion stopped
        return motion_stable, BoundaryType.SHARP

    def _detect_approach_withdraw_pattern(self, active_objects: Set[str]) -> bool:
        """
        Detect tool approaching then withdrawing from container
        """
        if len(self.frame_buffer) < 20:
            return False

        # Look for tool (pipette, hand) in active objects
        tool_objects = [
            obj for obj in self.frame_buffer[-1].objects
            if obj['class'] in ['pipette', 'hand'] and obj['id'] in active_objects
        ]

        if not tool_objects:
            return False

        # Check for approach-withdraw pattern in recent frames
        tool = tool_objects[0]
        tool_id = tool['id']

        heights = []
        for frame in list(self.frame_buffer)[-20:]:
            tool_in_frame = next(
                (o for o in frame.objects if o['id'] == tool_id),
                None
            )
            if tool_in_frame:
                heights.append(tool_in_frame['position'][2])  # Z coordinate

        if len(heights) < 15:
            return False

        # Check for dip pattern: high -> low -> high
        first_third = np.mean(heights[:5])
        middle = np.mean(heights[7:13])
        last_third = np.mean(heights[-5:])

        approach_withdraw = (first_third > middle) and (last_third > middle)
        return approach_withdraw and np.abs(first_third - last_third) < 0.05

    def _detect_oscillation_stop(self) -> bool:
        """
        Detect end of oscillatory motion (mixing)
        """
        if len(self.motion_history) < 20:
            return False

        recent = np.array(list(self.motion_history)[-20:])

        # Count zero-crossings (oscillations)
        mean = np.mean(recent)
        crossings = np.sum(np.diff(np.sign(recent - mean)) != 0)

        # Recent period should be stable
        last_5 = recent[-5:]
        is_stable = np.std(last_5) < self.stability_threshold

        # Previous period should have oscillations
        had_oscillations = crossings > 5

        return is_stable and had_oscillations

    def _detect_placement_stability(self, active_objects: Set[str]) -> bool:
        """
        Detect object placed and stable
        """
        if len(self.motion_history) < 15:
            return False

        # Check for placement event followed by stability
        motion = list(self.motion_history)

        # Early frames: motion (placement)
        early_motion = np.mean(motion[:5])

        # Recent frames: stable
        recent_motion = np.mean(motion[-10:])

        placed_and_stable = (
            early_motion > 0.2 and
            recent_motion < self.stability_threshold
        )

        return placed_and_stable

    def _extract_window_with_uncertainty(
        self,
        boundary_type: BoundaryType
    ) -> TemporalWindow:
        """
        Extract window with confidence scores for boundaries
        """
        frames = list(self.frame_buffer)

        if not frames:
            return None

        # Determine boundary confidence based on type and motion pattern
        start_confidence = self._calculate_boundary_confidence('start', boundary_type)
        end_confidence = self._calculate_boundary_confidence('end', boundary_type)

        # Collect active objects
        active_objects = set()
        for frame in frames:
            for obj in frame.objects:
                if obj.get('is_active', False):
                    active_objects.add(obj['id'])

        return TemporalWindow(
            frames=frames,
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            duration=frames[-1].timestamp - frames[0].timestamp,
            active_objects=active_objects,
            start_confidence=start_confidence,
            end_confidence=end_confidence,
            boundary_type=boundary_type
        )

    def _calculate_boundary_confidence(
        self,
        boundary: str,
        boundary_type: BoundaryType
    ) -> float:
        """
        Calculate confidence in boundary detection
        """
        if boundary_type == BoundaryType.SHARP:
            # Sharp boundaries -> high confidence
            return 0.90
        elif boundary_type == BoundaryType.GRADUAL:
            # Gradual boundaries -> medium confidence
            return 0.75
        else:  # CONTINUOUS
            # Continuous actions -> lower confidence in exact boundary
            return 0.60
