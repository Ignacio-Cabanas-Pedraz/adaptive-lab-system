"""
Rule-based Action Classifier
Classifies actions with nuanced confidence scoring
"""

import numpy as np
from typing import Dict, List, Optional, Set

from .data_structures import ActionType, TemporalWindow


class RuleBasedActionClassifier:
    """
    Rule-based action classification with nuanced confidence scoring
    Avoids confirmation bias by properly weighing visual evidence
    """

    def __init__(self):
        self.action_rules = {
            ActionType.TRANSFER: self._detect_transfer_pattern,
            ActionType.MIX: self._detect_mix_pattern,
            ActionType.HEAT: self._detect_heat_pattern,
            ActionType.COOL: self._detect_cool_pattern,
            ActionType.CENTRIFUGE: self._detect_centrifuge_pattern,
            ActionType.WAIT: self._detect_wait_pattern,
            ActionType.VORTEX: self._detect_vortex_pattern,
        }

    def classify(
        self,
        window: TemporalWindow,
        expected_action: Optional[ActionType] = None
    ) -> Dict:
        """
        Classify action type with proper confidence calculation
        """
        # Extract features
        features = self._extract_features(window)

        # Try each action type
        scores = {}
        for action_type, rule_func in self.action_rules.items():
            visual_confidence = rule_func(features)
            scores[action_type] = visual_confidence

        # Get best match based on visual evidence
        best_action = max(scores.items(), key=lambda x: x[1])
        detected_action = best_action[0]
        visual_confidence = best_action[1]

        # Calculate final confidence using template prior
        final_confidence = self._calculate_confidence(
            detected_action,
            expected_action,
            visual_confidence
        )

        # Check if detection is reliable
        matched_expectation = (detected_action == expected_action)

        return {
            'type': detected_action,
            'confidence': final_confidence,
            'visual_confidence': visual_confidence,
            'matched_expectation': matched_expectation,
            'scores': scores,  # For debugging
            'features': features  # For ML training later
        }

    def _calculate_confidence(
        self,
        detected_action: ActionType,
        expected_action: Optional[ActionType],
        visual_confidence: float
    ) -> float:
        """
        Combine visual confidence with template prior
        Avoids confirmation bias by not over-weighting template
        """
        # No expectation -> use visual confidence only
        if expected_action is None:
            return visual_confidence

        # Detection matches expectation
        if detected_action == expected_action:
            # Boost confidence, but not too much
            boost_factor = 1.2
            boosted = visual_confidence * boost_factor
            return min(0.98, boosted)  # Cap at 0.98 to avoid overconfidence

        # Detection doesn't match expectation
        else:
            # Could be genuine deviation or detection error
            if visual_confidence > 0.75:
                # Strong visual evidence against template -> trust vision
                penalty_factor = 0.9
            else:
                # Weak visual evidence -> more likely detection error
                penalty_factor = 0.7

            return visual_confidence * penalty_factor

    def _extract_features(self, window: TemporalWindow) -> Dict:
        """
        Extract features from temporal window
        """
        features = {
            'duration': window.duration,
            'object_count': len(window.active_objects),
            'tool_detected': self._has_tool(window),
            'tool_trajectory': self._get_tool_trajectory(window),
            'container_approaches': self._count_container_approaches(window),
            'oscillation_count': self._count_oscillations(window),
            'placement_events': self._detect_placement_events(window),
            'objects_present': self._get_object_classes(window),
            'motion_pattern': self._analyze_motion_pattern(window)
        }

        return features

    def _has_tool(self, window: TemporalWindow) -> bool:
        """Check if tool (pipette, hand) present"""
        for frame in window.frames:
            for obj in frame.objects:
                if obj['class'] in ['pipette', 'hand', 'tool']:
                    return True
        return False

    def _get_tool_trajectory(self, window: TemporalWindow) -> List[np.ndarray]:
        """Extract tool position over time"""
        trajectory = []
        tool_id = None

        # Find tool
        for frame in window.frames:
            for obj in frame.objects:
                if obj['class'] in ['pipette', 'hand']:
                    tool_id = obj['id']
                    break
            if tool_id:
                break

        if not tool_id:
            return []

        # Extract positions
        for frame in window.frames:
            for obj in frame.objects:
                if obj['id'] == tool_id:
                    trajectory.append(np.array(obj['position']))
                    break

        return trajectory

    def _count_container_approaches(self, window: TemporalWindow) -> int:
        """Count how many times tool approached a container"""
        trajectory = self._get_tool_trajectory(window)
        if len(trajectory) < 5:
            return 0

        # Detect approach events (tool getting closer to container)
        approaches = 0
        in_approach = False

        for i in range(len(trajectory) - 1):
            # Check Z coordinate (height)
            z_current = trajectory[i][2]
            z_next = trajectory[i+1][2]

            if z_next < z_current - 0.05:  # Descending
                if not in_approach:
                    approaches += 1
                    in_approach = True
            elif z_next > z_current + 0.05:  # Ascending
                in_approach = False

        return approaches

    def _count_oscillations(self, window: TemporalWindow) -> int:
        """Count oscillatory motions (for mixing detection)"""
        trajectory = self._get_tool_trajectory(window)
        if len(trajectory) < 10:
            return 0

        # Extract vertical component
        z_positions = np.array([pos[2] for pos in trajectory])

        # Detrend
        z_detrended = z_positions - np.mean(z_positions)

        # Count zero crossings
        crossings = np.sum(np.diff(np.sign(z_detrended)) != 0)

        # Each oscillation = 2 crossings
        return crossings // 2

    def _detect_placement_events(self, window: TemporalWindow) -> List[Dict]:
        """Detect objects being placed on surfaces"""
        events = []

        # Track each object's motion
        for obj_id in window.active_objects:
            positions = []
            for frame in window.frames:
                for obj in frame.objects:
                    if obj['id'] == obj_id:
                        positions.append(obj['position'])

            if len(positions) < 5:
                continue

            # Check for placement: motion -> stability
            early_positions = positions[:5]
            late_positions = positions[-5:]

            early_motion = np.std([p[2] for p in early_positions])
            late_motion = np.std([p[2] for p in late_positions])

            if early_motion > 0.05 and late_motion < 0.02:
                # Placed and stable
                duration = window.duration
                events.append({
                    'object': obj_id,
                    'duration': duration,
                    'target': 'unknown'
                })

        return events

    def _get_object_classes(self, window: TemporalWindow) -> Set[str]:
        """Get set of object classes present in window"""
        classes = set()
        for frame in window.frames:
            for obj in frame.objects:
                classes.add(obj['class'])
        return classes

    def _analyze_motion_pattern(self, window: TemporalWindow) -> str:
        """Classify overall motion pattern"""
        trajectory = self._get_tool_trajectory(window)
        if len(trajectory) < 10:
            return 'insufficient_data'

        # Calculate linearity
        start = trajectory[0]
        end = trajectory[-1]
        straight_line_distance = np.linalg.norm(end - start)

        path_distance = sum(
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory) - 1)
        )

        if path_distance < 1e-6:
            return 'stationary'

        linearity = straight_line_distance / path_distance

        if linearity > 0.8:
            return 'linear'
        elif linearity > 0.5:
            return 'curved'
        else:
            return 'oscillatory'

    # Rule functions

    def _detect_transfer_pattern(self, features: Dict) -> float:
        """Detect transfer action"""
        if not features['tool_detected']:
            return 0.0

        score = 0.0

        # Linear motion pattern
        if features['motion_pattern'] == 'linear':
            score += 0.3

        # Multiple container approaches
        approaches = features['container_approaches']
        if approaches >= 2:
            score += 0.5
        elif approaches == 1:
            score += 0.2

        # Reasonable duration (2-10 seconds)
        if 2.0 < features['duration'] < 10.0:
            score += 0.2

        return min(1.0, score)

    def _detect_mix_pattern(self, features: Dict) -> float:
        """Detect mixing action"""
        if not features['tool_detected']:
            return 0.0

        score = 0.0

        # Oscillatory motion
        if features['motion_pattern'] == 'oscillatory':
            score += 0.4

        # Multiple oscillations
        oscillations = features['oscillation_count']
        if oscillations >= 5:
            score += 0.5
        elif oscillations >= 3:
            score += 0.3

        # Reasonable duration
        if 3.0 < features['duration'] < 30.0:
            score += 0.1

        return min(1.0, score)

    def _detect_heat_pattern(self, features: Dict) -> float:
        """Detect heating action"""
        score = 0.0

        # Heat source present
        if 'heater' in features['objects_present']:
            score += 0.5

        # Placement event
        if len(features['placement_events']) > 0:
            score += 0.3

            # Long duration placement
            max_duration = max(e['duration'] for e in features['placement_events'])
            if max_duration > 5.0:
                score += 0.2

        return min(1.0, score)

    def _detect_cool_pattern(self, features: Dict) -> float:
        """Detect cooling action"""
        score = 0.0

        # Ice bucket or cooler present
        if 'ice_bucket' in features['objects_present'] or 'cooler' in features['objects_present']:
            score += 0.5

        # Placement event
        if len(features['placement_events']) > 0:
            score += 0.4

        return min(1.0, score)

    def _detect_centrifuge_pattern(self, features: Dict) -> float:
        """Detect centrifugation"""
        score = 0.0

        # Centrifuge present
        if 'centrifuge' in features['objects_present']:
            score += 0.7

        # Placement event
        if len(features['placement_events']) > 0:
            score += 0.3

        return min(1.0, score)

    def _detect_wait_pattern(self, features: Dict) -> float:
        """Detect waiting/incubation"""
        score = 0.0

        # Stationary pattern
        if features['motion_pattern'] == 'stationary':
            score += 0.5

        # Long duration
        if features['duration'] > 10.0:
            score += 0.3

        # Placement event with long duration
        if len(features['placement_events']) > 0:
            max_duration = max(e['duration'] for e in features['placement_events'])
            if max_duration > 10.0:
                score += 0.2

        return min(1.0, score)

    def _detect_vortex_pattern(self, features: Dict) -> float:
        """Detect vortexing"""
        score = 0.0

        # Vortex mixer present
        if 'vortex' in features['objects_present']:
            score += 0.6

        # High frequency oscillations
        if features['oscillation_count'] > 10:
            score += 0.3

        # Short duration
        if 2.0 < features['duration'] < 15.0:
            score += 0.1

        return min(1.0, score)
