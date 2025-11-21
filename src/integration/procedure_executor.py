"""
Runtime procedure execution orchestration
Main integration layer between template, TEP, and video processing
"""

import json
import time
from typing import Dict, List, Optional
from pathlib import Path

from ..tep.data_structures import ActionType, FrameData, TEPEvent
from ..tep.temporal_event_parser import TemporalEventParser
from ..procedure.template_generator import ProcedureTemplateGenerator
from .frame_converter import FrameConverter, BatchFrameConverter


class ProcedureContext:
    """
    Provides procedure template context to TEP
    """

    def __init__(self, template: Dict):
        self.template = template
        self.current_step_number = 1
        self.total_steps = len(template.get('steps', []))

    def get_current_step(self) -> Dict:
        """Get current step from template"""
        steps = self.template.get('steps', [])
        if 0 < self.current_step_number <= len(steps):
            return steps[self.current_step_number - 1]
        return {'expected_action': 'wait', 'description': 'No step'}

    def get_step(self, step_number: int) -> Optional[Dict]:
        """Get specific step by number"""
        steps = self.template.get('steps', [])
        if 1 <= step_number <= len(steps):
            return steps[step_number - 1]
        return None

    def advance_step(self):
        """Move to next step"""
        if self.current_step_number < self.total_steps:
            self.current_step_number += 1

    def jump_to_step(self, step_number: int):
        """Jump to specific step"""
        if 1 <= step_number <= self.total_steps:
            self.current_step_number = step_number


class ProcedureExecutor:
    """
    Main runtime orchestrator
    Connects video processing -> TEP -> execution log
    """

    def __init__(
        self,
        template: Dict,
        frame_width: int = 1920,
        frame_height: int = 1080
    ):
        # Create procedure context
        self.context = ProcedureContext(template)

        # Initialize TEP
        self.tep = TemporalEventParser(self.context)

        # Initialize frame converter
        self.frame_converter = FrameConverter(frame_width, frame_height)

        # Execution state
        self.events = []
        self.start_time = None
        self.is_running = False

    def start(self):
        """Start procedure execution"""
        self.start_time = time.time()
        self.is_running = True
        self.events = []

    def stop(self):
        """Stop procedure execution"""
        self.is_running = False

    def process_yolo_result(
        self,
        yolo_result: Dict,
        frame_number: int,
        timestamp: float
    ) -> Optional[TEPEvent]:
        """
        Process single YOLO detection result through TEP

        Args:
            yolo_result: YOLO detection output
            frame_number: Frame number
            timestamp: Frame timestamp

        Returns:
            TEPEvent if action detected, None otherwise
        """
        if not self.is_running:
            return None

        # Convert to FrameData
        frame_data = self.frame_converter.convert(
            yolo_result=yolo_result,
            frame_number=frame_number,
            timestamp=timestamp
        )

        # Process through TEP
        event = self.tep.process_frame(frame_data)

        if event:
            self.events.append(event)

            # Auto-advance step if matched
            if event.matched_expectation:
                self.context.advance_step()

        return event

    def process_video_results(
        self,
        results: List[Dict],
        fps: float = 30.0
    ) -> List[TEPEvent]:
        """
        Process entire video through TEP

        Args:
            results: List of YOLO detection results
            fps: Video frame rate

        Returns:
            List of detected events
        """
        self.start()

        # Convert all frames
        batch_converter = BatchFrameConverter(
            self.frame_converter.frame_width,
            self.frame_converter.frame_height
        )
        frame_data_list = batch_converter.convert_batch(results, fps)

        # Process each frame
        events = []
        for frame_data in frame_data_list:
            event = self.tep.process_frame(frame_data)
            if event:
                events.append(event)
                self.events.append(event)

                # Auto-advance step if matched
                if event.matched_expectation:
                    self.context.advance_step()

        self.stop()
        return events

    def get_execution_log(self) -> Dict:
        """
        Generate execution log from events

        Returns:
            Dictionary with execution summary and events
        """
        # Calculate statistics
        total_events = len(self.events)
        matched_events = sum(1 for e in self.events if e.matched_expectation)
        avg_confidence = (
            sum(e.confidence for e in self.events) / total_events
            if total_events > 0 else 0.0
        )

        # Get completed steps
        completed_steps = set()
        for event in self.events:
            if event.matched_expectation and event.step_number:
                completed_steps.add(event.step_number)

        return {
            'procedure_id': self.context.template.get('template_id', 'unknown'),
            'procedure_title': self.context.template.get('title', 'Unknown'),
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_steps': self.context.total_steps,
            'completed_steps': len(completed_steps),
            'completion_rate': len(completed_steps) / max(self.context.total_steps, 1),
            'statistics': {
                'total_events': total_events,
                'matched_events': matched_events,
                'match_rate': matched_events / max(total_events, 1),
                'average_confidence': avg_confidence
            },
            'events': [event.to_dict() for event in self.events]
        }

    def export_execution_log(self, output_path: str):
        """
        Export execution log to JSON file

        Args:
            output_path: Path to save JSON file
        """
        log = self.get_execution_log()

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)


def create_executor_from_text(
    step_descriptions: List[str],
    title: str = "Lab Procedure",
    user_id: str = "system"
) -> ProcedureExecutor:
    """
    Convenience function to create executor from text steps

    Args:
        step_descriptions: List of step description strings
        title: Procedure title
        user_id: User ID

    Returns:
        Configured ProcedureExecutor
    """
    # Generate template
    generator = ProcedureTemplateGenerator()
    template = generator.generate_template(
        title=title,
        user_id=user_id,
        step_descriptions=step_descriptions
    )

    # Create executor
    return ProcedureExecutor(template)
