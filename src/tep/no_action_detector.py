"""
No-Action Detector
Detects and classifies periods of inactivity
"""

import time
import numpy as np
from typing import Dict, Optional

from .data_structures import ActionType


class NoActionDetector:
    """
    Detects and classifies periods of inactivity
    Distinguishes between expected waiting and unexpected pauses
    """

    def __init__(self):
        self.inactivity_threshold = 5.0  # seconds
        self.workspace_exit_threshold = 10.0  # seconds
        self.last_action_time = time.time()

    def check_inactivity(
        self,
        current_time: float,
        current_step: Dict,
        imu_data: Dict
    ) -> Optional[Dict]:
        """
        Check if current inactivity is expected or problematic
        """
        time_since_action = current_time - self.last_action_time

        # Short inactivity -> normal
        if time_since_action < self.inactivity_threshold:
            return None

        # Expected wait step
        expected_action = current_step.get('expected_action', '')
        if expected_action == 'wait' or expected_action == ActionType.WAIT:
            return {
                'type': ActionType.WAIT,
                'confidence': 0.95,
                'validated': True,
                'message': f"Waiting as expected ({time_since_action:.0f}s)"
            }

        # Check IMU for workspace exit
        if self._detect_workspace_exit(imu_data, time_since_action):
            return {
                'type': ActionType.WORKSPACE_EXIT,
                'confidence': 0.90,
                'validated': False,
                'action': 'pause_procedure',
                'message': "User appears to have left workspace"
            }

        # Unexpected pause
        if time_since_action > self.workspace_exit_threshold:
            return {
                'type': ActionType.UNEXPECTED_PAUSE,
                'confidence': 0.75,
                'validated': False,
                'action': 'check_if_user_needs_help',
                'message': f"No activity for {time_since_action:.0f}s. Need help?"
            }

        return None

    def _detect_workspace_exit(self, imu_data: Dict, inactivity_duration: float) -> bool:
        """
        Detect if user has left the workspace
        Uses IMU motion patterns
        """
        if not imu_data:
            return False

        # Check for significant head movement away from workspace
        accel = imu_data.get('accelerometer', [0, 0, 0])
        accel_magnitude = np.linalg.norm(accel)

        # Walking typically shows oscillation with magnitude 0.5-2.0 m/s²
        is_walking = 0.5 < accel_magnitude < 2.0

        # Long inactivity + walking -> probably left
        return is_walking and inactivity_duration > 5.0

    def reset_timer(self):
        """Reset inactivity timer after action detected"""
        self.last_action_time = time.time()
