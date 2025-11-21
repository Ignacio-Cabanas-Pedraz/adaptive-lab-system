"""
Deviation Handler
Handles cases where detected actions don't match template expectations
"""

import time
from collections import deque
from typing import Dict, Optional

from .data_structures import ActionType, MismatchType, TEPEvent


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
        procedure_context
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
            'step': current_step.get('step_number', 0),
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

        # Pattern 3: Multiple recent mismatches -> user is lost
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
        procedure_context
    ) -> Optional[int]:
        """
        Check if detected action matches a future step (within next 3)
        """
        current_step = procedure_context.current_step_number

        for i in range(1, 4):  # Look ahead 3 steps
            step_num = current_step + i
            if step_num <= procedure_context.total_steps:
                step = procedure_context.get_step(step_num)
                if step and step.get('expected_action') == action.value:
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
