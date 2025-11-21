"""
Shared data structures for TEP system
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import numpy as np


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
    SHARP = "sharp"
    GRADUAL = "gradual"
    CONTINUOUS = "continuous"


@dataclass
class FrameData:
    """
    Single frame of vision data
    Converted from YOLO/SAM/CLIP output
    """
    timestamp: float
    frame_number: int
    objects: List[Dict]  # [{id, class, bbox, position, confidence}]
    imu_data: Optional[Dict] = None  # For future smart glasses
    workspace_id: str = "default"

    def __post_init__(self):
        """Validate frame data"""
        if not isinstance(self.objects, list):
            raise ValueError("objects must be a list")

        for obj in self.objects:
            required_keys = ['id', 'class', 'position']
            if not all(k in obj for k in required_keys):
                raise ValueError(f"Object missing required keys: {required_keys}")


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

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        if self.duration <= 0:
            return 0.0
        return self.frame_count / self.duration


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

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'action_type': self.action_type.value,
            'confidence': self.confidence,
            'step_number': self.step_number,
            'step_description': self.step_description,
            'expected_action': self.expected_action.value if self.expected_action else None,
            'matched_expectation': self.matched_expectation,
            'duration': self.duration,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'objects': list(self.objects),
            'workspace': self.workspace,
            'validated': self.validated,
            'warnings': self.warnings,
            'parameters': self.parameters
        }


@dataclass
class ProcedureTemplate:
    """Complete procedure template"""
    template_id: str
    title: str
    version: str
    created_by: str
    created_at: str
    steps: List[Dict]
    post_procedure: List[str]
    metadata: Dict = field(default_factory=dict)

    def get_step(self, step_number: int) -> Optional[Dict]:
        """Get step by number (1-indexed)"""
        if 1 <= step_number <= len(self.steps):
            return self.steps[step_number - 1]
        return None

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'template_id': self.template_id,
            'title': self.title,
            'version': self.version,
            'created_by': self.created_by,
            'created_at': self.created_at,
            'metadata': self.metadata,
            'steps': self.steps,
            'post_procedure': self.post_procedure
        }
