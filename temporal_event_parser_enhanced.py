"""
Temporal Event Parser (TEP) - Enhanced Implementation
Template-Guided Action Validation System

This implementation incorporates:
1. Nuanced confidence scoring (avoid confirmation bias)
2. Graceful handling of user deviations
3. No-action detection with context awareness
4. Multi-object interaction patterns
5. Temporal boundary uncertainty tracking
"""

import numpy as np
import networkx as nx
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import time
import uuid


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ActionType(Enum):
    """Supported action types"""
    TRANSFER = "transfer"
    MIX = "mix"
    HEAT = "heat"
    COOL = "cool"
    CENTRIFUGE = "centrifuge"
    WAIT = "wait"
    MEASURE = "measure"
    VORTEX = "vortex"
    OPEN_CLOSE = "open_close"
    DISCARD = "discard"
    NONE = "none"
    WORKSPACE_EXIT = "workspace_exit"
    UNEXPECTED_PAUSE = "unexpected_pause"


class MismatchType(Enum):
    """Types of template-execution mismatches"""
    SKIP_AHEAD = "skip_ahead"
    EXTRA_STEP = "extra_step"
    LOST_SYNC = "lost_sync"
    MINOR_MISMATCH = "minor_mismatch"
    USER_CORRECTION = "user_correction"


class BoundaryType(Enum):
    """Action boundary characteristics"""
    SHARP = "sharp"  # Clear start/end (e.g., transfer)
    GRADUAL = "gradual"  # Fuzzy boundaries (e.g., mixing)
    CONTINUOUS = "continuous"  # Ongoing (e.g., heating)


@dataclass
class FrameData:
    """Single frame of vision data"""
    timestamp: float
    objects: List[Dict]  # Detected objects with positions, classes
    imu_data: Dict  # Accelerometer, gyroscope readings
    workspace_id: str


@dataclass
class TemporalWindow:
    """Window of frames for action detection"""
    frames: List[FrameData]
    start_time: float
    end_time: float
    duration: float
    active_objects: Set[str]
    start_confidence: float = 0.0
    end_confidence: float = 0.0
    boundary_type: BoundaryType = BoundaryType.SHARP


@dataclass
class TEPEvent:
    """Detected action event with full context"""
    event_id: str
    timestamp: float
    action_type: ActionType
    confidence: float
    
    # Template context
    step_number: Optional[int] = None
    step_description: Optional[str] = None
    expected_action: Optional[ActionType] = None
    matched_expectation: bool = False
    
    # Timing
    duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Objects and workspace
    objects: Set[str] = field(default_factory=set)
    workspace: str = ""
    
    # Validation
    validated: bool = False
    visual_confidence: float = 0.0
    mismatch_type: Optional[MismatchType] = None
    user_corrected: bool = False
    
    # Additional metadata
    parameters: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # For ML training
    features: Optional[Dict] = None
    window_data: Optional[TemporalWindow] = None


# ============================================================================
# TEMPORAL WINDOW MANAGER (Enhanced)
# ============================================================================

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
        # Simple heuristic: tool height decreased then increased
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
        
        # Check for dip pattern: high → low → high
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
        prev_15 = recent[:15]
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
            # Sharp boundaries → high confidence
            return 0.90
        elif boundary_type == BoundaryType.GRADUAL:
            # Gradual boundaries → medium confidence
            return 0.75
        else:  # CONTINUOUS
            # Continuous actions → lower confidence in exact boundary
            return 0.60


# ============================================================================
# INTERACTION GRAPH BUILDER (Enhanced)
# ============================================================================

class InteractionGraphBuilder:
    """
    Builds spatial relationship graphs with multi-object pattern detection
    """
    
    def __init__(self, proximity_threshold: float = 0.15):
        self.proximity_threshold = proximity_threshold
    
    def build_graph(
        self,
        frame_data: FrameData,
        expected_objects: List[str]
    ) -> nx.Graph:
        """
        Create interaction graph focused on expected objects
        """
        G = nx.Graph()
        
        # Filter to relevant objects
        relevant_objects = [
            obj for obj in frame_data.objects
            if (obj['id'] in expected_objects or 
                obj['class'] in ['pipette', 'hand', 'tool'])
        ]
        
        # Add nodes
        for obj in relevant_objects:
            G.add_node(
                obj['id'],
                position=obj['position'],
                obj_class=obj['class'],
                is_expected=obj['id'] in expected_objects
            )
        
        # Add edges for proximal objects
        for i, obj1 in enumerate(relevant_objects):
            for obj2 in relevant_objects[i+1:]:
                distance = self._compute_distance(obj1, obj2)
                
                if distance < self.proximity_threshold:
                    G.add_edge(
                        obj1['id'],
                        obj2['id'],
                        distance=distance,
                        alignment=self._compute_alignment(obj1, obj2)
                    )
        
        return G
    
    def detect_multi_object_patterns(
        self,
        graph_sequence: List[nx.Graph]
    ) -> Dict[str, bool]:
        """
        Detect common multi-object interaction patterns
        """
        patterns = {
            'serial_transfer': self._detect_serial_transfer(graph_sequence),
            'parallel_pipetting': self._detect_parallel_pipetting(graph_sequence),
            'wash_cycle': self._detect_wash_cycle(graph_sequence)
        }
        
        return patterns
    
    def _detect_serial_transfer(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Tool visits A → B → C sequentially
        Common in dilution series
        """
        if len(graph_sequence) < 10:
            return False
        
        # Track tool proximity to containers over time
        tool_nodes = self._find_tools_in_graphs(graph_sequence)
        if not tool_nodes:
            return False
        
        tool = tool_nodes[0]
        
        # Track which containers tool visited
        visited_containers = []
        current_container = None
        
        for graph in graph_sequence:
            if not graph.has_node(tool):
                continue
            
            # Find containers near tool
            neighbors = [
                n for n in graph.neighbors(tool)
                if graph.nodes[n].get('obj_class') in ['tube', 'beaker', 'well']
            ]
            
            if neighbors and neighbors[0] != current_container:
                visited_containers.append(neighbors[0])
                current_container = neighbors[0]
        
        # Serial transfer = tool visited 3+ distinct containers
        return len(set(visited_containers)) >= 3
    
    def _detect_parallel_pipetting(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Tool rapidly alternates between 2 containers
        Common in multichannel pipetting
        """
        if len(graph_sequence) < 5:
            return False
        
        tool_nodes = self._find_tools_in_graphs(graph_sequence)
        if not tool_nodes:
            return False
        
        tool = tool_nodes[0]
        
        # Track alternation pattern
        container_sequence = []
        for graph in graph_sequence:
            if not graph.has_node(tool):
                continue
            
            neighbors = [
                n for n in graph.neighbors(tool)
                if graph.nodes[n].get('obj_class') in ['tube', 'beaker', 'well']
            ]
            
            if neighbors:
                container_sequence.append(neighbors[0])
        
        # Check for ABAB pattern
        if len(container_sequence) < 4:
            return False
        
        # Count alternations
        alternations = 0
        for i in range(len(container_sequence) - 1):
            if container_sequence[i] != container_sequence[i+1]:
                alternations += 1
        
        return alternations >= 3
    
    def _detect_wash_cycle(self, graph_sequence: List[nx.Graph]) -> bool:
        """
        Pattern: Container moved to wash station, held, then removed
        Common in plate washing
        """
        # TODO: Implement wash cycle detection
        # Requires identifying wash station and dwell time
        return False
    
    def _find_tools_in_graphs(self, graph_sequence: List[nx.Graph]) -> List[str]:
        """
        Find tool nodes across graph sequence
        """
        tools = set()
        for graph in graph_sequence:
            for node, data in graph.nodes(data=True):
                if data.get('obj_class') in ['pipette', 'hand', 'tool']:
                    tools.add(node)
        return list(tools)
    
    def _compute_distance(self, obj1: Dict, obj2: Dict) -> float:
        """
        Euclidean distance between object centers
        """
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        return np.linalg.norm(pos1 - pos2)
    
    def _compute_alignment(self, obj1: Dict, obj2: Dict) -> float:
        """
        Alignment score (0-1) based on relative orientation
        1.0 = perfectly aligned, 0.0 = perpendicular
        """
        # Simplified: use position vector
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        
        direction = pos2 - pos1
        if np.linalg.norm(direction) < 1e-6:
            return 0.0
        
        direction = direction / np.linalg.norm(direction)
        
        # Alignment with vertical axis (common for pipetting)
        vertical = np.array([0, 0, 1])
        alignment = np.abs(np.dot(direction, vertical))
        
        return alignment


# ============================================================================
# ACTION TYPE CLASSIFIER (Enhanced with Better Confidence)
# ============================================================================

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
        # No expectation → use visual confidence only
        if expected_action is None:
            return visual_confidence
        
        # Detection matches expectation
        if detected_action == expected_action:
            # Boost confidence, but not too much
            # We still want to catch cases where user deviated
            boost_factor = 1.2
            boosted = visual_confidence * boost_factor
            return min(0.98, boosted)  # Cap at 0.98 to avoid overconfidence
        
        # Detection doesn't match expectation
        else:
            # Could be genuine deviation or detection error
            # Don't penalize too harshly if visual confidence is high
            if visual_confidence > 0.75:
                # Strong visual evidence against template → trust vision
                penalty_factor = 0.9
            else:
                # Weak visual evidence → more likely detection error
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
            # Simplified: check Z coordinate (height)
            # Lower = closer to container
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
            
            # Check for placement: motion → stability
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
                    'target': 'unknown'  # Would need more context
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


# ============================================================================
# DEVIATION HANDLER (New Component)
# ============================================================================

class DeviationHandler:
    """
    Handles cases where detected actions don't match template expectations
    Implements graceful degradation and smart user interaction
    """
    
    def __init__(self):
        self.deviation_buffer = deque(maxlen=10)
        self.user_corrections = []
    
    def handle_mismatch(
        self,
        detected_action: ActionType,
        expected_action: ActionType,
        current_step: Dict,
        procedure_context: 'ProcedureContext'
    ) -> Dict:
        """
        Smart handling of template-execution mismatch
        
        Returns action recommendation:
        - soft_warning: Log but don't interrupt
        - prompt_user: Ask for confirmation
        - skip_ahead: User jumped to future step
        - extra_step: User added step not in template
        """
        # Log mismatch
        self.deviation_buffer.append({
            'detected': detected_action,
            'expected': expected_action,
            'step': current_step['number'],
            'timestamp': time.time()
        })
        
        # Pattern 1: Detected action matches a future step
        future_match = self._find_matching_future_step(
            detected_action,
            procedure_context
        )
        
        if future_match:
            return {
                'type': MismatchType.SKIP_AHEAD,
                'action': 'soft_warning',
                'matched_step': future_match,
                'message': f"Detected Step {future_match} action. Did you skip steps?",
                'severity': 'medium'
            }
        
        # Pattern 2: Detected action is reasonable but not in template
        if self._is_reasonable_action(detected_action):
            return {
                'type': MismatchType.EXTRA_STEP,
                'action': 'log_and_continue',
                'suggest_template_update': True,
                'message': f"Detected {detected_action.value} (not in template)",
                'severity': 'low'
            }
        
        # Pattern 3: Multiple recent mismatches → user is lost
        if self._check_lost_sync():
            return {
                'type': MismatchType.LOST_SYNC,
                'action': 'prompt_user',
                'message': "I'm having trouble following. What step are you on?",
                'severity': 'high'
            }
        
        # Pattern 4: Minor mismatch, log with flag
        return {
            'type': MismatchType.MINOR_MISMATCH,
            'action': 'log_with_flag',
            'message': f"Expected {expected_action.value}, detected {detected_action.value}",
            'severity': 'low'
        }
    
    def _find_matching_future_step(
        self,
        action: ActionType,
        procedure_context: 'ProcedureContext'
    ) -> Optional[int]:
        """
        Check if detected action matches a future step (within next 3)
        """
        current_step = procedure_context.current_step_number
        
        for i in range(1, 4):  # Look ahead 3 steps
            step_num = current_step + i
            if step_num <= procedure_context.total_steps:
                step = procedure_context.get_step(step_num)
                if step['expected_action'] == action:
                    return step_num
        
        return None
    
    def _is_reasonable_action(self, action: ActionType) -> bool:
        """
        Check if action is reasonable lab work (vs. noise)
        """
        reasonable = [
            ActionType.TRANSFER,
            ActionType.MIX,
            ActionType.HEAT,
            ActionType.COOL,
            ActionType.VORTEX,
            ActionType.WAIT
        ]
        return action in reasonable
    
    def _check_lost_sync(self) -> bool:
        """
        Check if multiple recent mismatches suggest user is lost
        """
        if len(self.deviation_buffer) < 3:
            return False
        
        # Check recent 3 deviations
        recent = list(self.deviation_buffer)[-3:]
        
        # All are mismatches?
        all_mismatch = all(
            d['detected'] != d['expected']
            for d in recent
        )
        
        # Happened within 60 seconds?
        time_span = recent[-1]['timestamp'] - recent[0]['timestamp']
        recent_span = time_span < 60.0
        
        return all_mismatch and recent_span
    
    def record_user_correction(
        self,
        event: TEPEvent,
        correct_action: ActionType
    ):
        """
        Record user correction for ML training
        """
        self.user_corrections.append({
            'event_id': event.event_id,
            'detected': event.action_type,
            'corrected_to': correct_action,
            'features': event.features,
            'timestamp': time.time()
        })


# ============================================================================
# NO-ACTION DETECTOR (New Component)
# ============================================================================

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
        
        # Short inactivity → normal
        if time_since_action < self.inactivity_threshold:
            return None
        
        # Expected wait step
        if current_step['expected_action'] == ActionType.WAIT:
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
        # Simplified: check if accelerometer shows walking pattern
        accel = imu_data.get('accelerometer', [0, 0, 0])
        accel_magnitude = np.linalg.norm(accel)
        
        # Walking typically shows 1-2 Hz oscillation with magnitude 0.5-2.0 m/s²
        is_walking = 0.5 < accel_magnitude < 2.0
        
        # Long inactivity + walking → probably left
        return is_walking and inactivity_duration > 5.0
    
    def reset_timer(self):
        """Reset inactivity timer after action detected"""
        self.last_action_time = time.time()


# ============================================================================
# PROCEDURE CONTEXT (Helper)
# ============================================================================

class ProcedureContext:
    """
    Provides procedure template context to TEP
    """
    
    def __init__(self, template: Dict):
        self.template = template
        self.current_step_number = 1
        self.total_steps = len(template['steps'])
    
    def get_current_step(self) -> Dict:
        """Get current step from template"""
        return self.template['steps'][self.current_step_number - 1]
    
    def get_step(self, step_number: int) -> Dict:
        """Get specific step by number"""
        if 1 <= step_number <= self.total_steps:
            return self.template['steps'][step_number - 1]
        return None
    
    def advance_step(self):
        """Move to next step"""
        if self.current_step_number < self.total_steps:
            self.current_step_number += 1
    
    def jump_to_step(self, step_number: int):
        """Jump to specific step"""
        if 1 <= step_number <= self.total_steps:
            self.current_step_number = step_number


# ============================================================================
# MAIN TEP ORCHESTRATOR
# ============================================================================

class TemporalEventParser:
    """
    Main TEP orchestrator
    Coordinates all components with enhanced error handling
    """
    
    def __init__(self, procedure_context: ProcedureContext):
        self.procedure_context = procedure_context
        
        # Components
        self.window_manager = TemporalWindowManager()
        self.graph_builder = InteractionGraphBuilder()
        self.classifier = RuleBasedActionClassifier()
        self.deviation_handler = DeviationHandler()
        self.no_action_detector = NoActionDetector()
        
        # State
        self.event_buffer = []
        self.last_event_time = time.time()
    
    def process_frame(self, frame_data: FrameData) -> Optional[TEPEvent]:
        """
        Process single frame, return event if action detected
        """
        current_step = self.procedure_context.get_current_step()
        
        # Check for inactivity
        inactivity_result = self.no_action_detector.check_inactivity(
            frame_data.timestamp,
            current_step,
            frame_data.imu_data
        )
        
        if inactivity_result:
            # Create event for inactivity
            event = self._create_inactivity_event(inactivity_result)
            return event
        
        # Update window manager
        window = self.window_manager.update(frame_data, current_step)
        
        if window is None:
            return None  # No action boundary detected yet
        
        # Action completed, classify it
        expected_action = ActionType(current_step['expected_action'])
        
        classification = self.classifier.classify(
            window,
            expected_action
        )
        
        detected_action = classification['type']
        
        # Check for mismatch
        if detected_action != expected_action:
            mismatch_response = self.deviation_handler.handle_mismatch(
                detected_action,
                expected_action,
                current_step,
                self.procedure_context
            )
            
            # Create event with mismatch info
            event = self._create_event(
                classification,
                window,
                current_step,
                mismatch_response
            )
        else:
            # Matched expectation
            event = self._create_event(
                classification,
                window,
                current_step
            )
        
        # Reset no-action detector
        self.no_action_detector.reset_timer()
        
        # Store event
        self.event_buffer.append(event)
        self.last_event_time = frame_data.timestamp
        
        return event
    
    def _create_event(
        self,
        classification: Dict,
        window: TemporalWindow,
        current_step: Dict,
        mismatch_info: Optional[Dict] = None
    ) -> TEPEvent:
        """
        Create structured event from classification
        """
        detected_action = classification['type']
        expected_action = ActionType(current_step['expected_action'])
        matched = classification['matched_expectation']
        
        event = TEPEvent(
            event_id=str(uuid.uuid4()),
            timestamp=window.end_time,
            action_type=detected_action,
            confidence=classification['confidence'],
            visual_confidence=classification['visual_confidence'],
            
            # Template context
            step_number=current_step['number'],
            step_description=current_step['description'],
            expected_action=expected_action,
            matched_expectation=matched,
            
            # Timing
            duration=window.duration,
            start_time=window.start_time,
            end_time=window.end_time,
            
            # Objects
            objects=window.active_objects,
            workspace=window.frames[0].workspace_id if window.frames else "",
            
            # Validation
            validated=matched and classification['confidence'] > 0.80,
            
            # For ML training
            features=classification.get('features'),
            window_data=window
        )
        
        # Add mismatch information if present
        if mismatch_info:
            event.mismatch_type = mismatch_info['type']
            event.warnings.append(mismatch_info['message'])
        
        # Add low confidence warning
        if classification['confidence'] < 0.70:
            event.warnings.append(
                f"Low confidence ({classification['confidence']:.2f})"
            )
        
        return event
    
    def _create_inactivity_event(self, inactivity_result: Dict) -> TEPEvent:
        """
        Create event for inactivity periods
        """
        return TEPEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            action_type=inactivity_result['type'],
            confidence=inactivity_result['confidence'],
            validated=inactivity_result['validated'],
            warnings=[inactivity_result.get('message', '')]
        )
    
    def get_training_data(self) -> List[Dict]:
        """
        Export data for ML training
        """
        return [
            {
                'event_id': event.event_id,
                'features': event.features,
                'label': event.action_type.value,
                'confidence': event.confidence,
                'user_corrected': event.user_corrected
            }
            for event in self.event_buffer
            if event.features is not None
        ]


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example template
    template = {
        'steps': [
            {
                'number': 1,
                'description': 'Add 200µL compound A to solution B',
                'expected_action': 'transfer',
                'expected_objects': ['compound_A_tube', 'solution_B_beaker'],
                'parameters': {
                    'volume': '200µL',
                    'source': 'compound A',
                    'destination': 'solution B'
                }
            },
            {
                'number': 2,
                'description': 'Mix by pipetting 10 times',
                'expected_action': 'mix',
                'expected_objects': ['solution_B_beaker'],
                'parameters': {
                    'repetitions': 10
                }
            }
        ]
    }
    
    # Initialize
    procedure_context = ProcedureContext(template)
    tep = TemporalEventParser(procedure_context)
    
    # Simulate processing frames
    print("TEP Enhanced Implementation Ready")
    print(f"Template loaded: {len(template['steps'])} steps")
    print(f"Current step: {procedure_context.get_current_step()['description']}")
