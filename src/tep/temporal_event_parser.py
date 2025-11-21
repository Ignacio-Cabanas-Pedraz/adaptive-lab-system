"""
Temporal Event Parser - Main orchestrator
Coordinates all components with enhanced error handling
"""

import time
import uuid
from typing import Dict, List, Optional

from .data_structures import (
    ActionType, FrameData, TEPEvent, TemporalWindow, MismatchType
)
from .window_manager import TemporalWindowManager
from .graph_builder import InteractionGraphBuilder
from .action_classifier import RuleBasedActionClassifier
from .deviation_handler import DeviationHandler
from .no_action_detector import NoActionDetector


class TemporalEventParser:
    """
    Main TEP orchestrator
    Coordinates all components with enhanced error handling
    """

    def __init__(self, procedure_context):
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
        imu_data = frame_data.imu_data if frame_data.imu_data else {}
        inactivity_result = self.no_action_detector.check_inactivity(
            frame_data.timestamp,
            current_step,
            imu_data
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
        expected_action_str = current_step.get('expected_action', 'wait')
        try:
            expected_action = ActionType(expected_action_str)
        except ValueError:
            expected_action = ActionType.WAIT

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
        expected_action_str = current_step.get('expected_action', 'wait')
        try:
            expected_action = ActionType(expected_action_str)
        except ValueError:
            expected_action = ActionType.WAIT

        matched = classification['matched_expectation']

        event = TEPEvent(
            event_id=str(uuid.uuid4()),
            timestamp=window.end_time,
            action_type=detected_action,
            confidence=classification['confidence'],
            visual_confidence=classification['visual_confidence'],

            # Template context
            step_number=current_step.get('step_number'),
            step_description=current_step.get('description'),
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
