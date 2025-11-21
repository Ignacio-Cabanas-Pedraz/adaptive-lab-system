#!/usr/bin/env python3
"""
Test TEP on video with procedure template
End-to-end test script for the Temporal Event Parser system
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.procedure.template_generator import ProcedureTemplateGenerator
from src.integration.procedure_executor import ProcedureExecutor, create_executor_from_text
from src.tep.data_structures import FrameData


def load_procedure_from_file(filepath: str) -> list:
    """
    Load procedure steps from text file

    Args:
        filepath: Path to procedure text file

    Returns:
        List of step descriptions
    """
    steps = []
    current_step = ""

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Step"):
                # Save previous step if exists
                if current_step:
                    steps.append(current_step)
                current_step = ""
            elif line:
                # Add to current step
                if current_step:
                    current_step += " " + line
                else:
                    current_step = line

    # Don't forget last step
    if current_step:
        steps.append(current_step)

    return steps


def create_simulated_video_results(num_frames: int = 100) -> list:
    """
    Create simulated YOLO detection results for testing

    In real usage, these would come from process_video.py

    Args:
        num_frames: Number of frames to simulate

    Returns:
        List of simulated YOLO results
    """
    import random

    results = []

    # Define object classes that might appear
    object_classes = [
        'tube', 'beaker', 'pipette', 'hand', 'vortex',
        'centrifuge', 'ice_bucket', 'heater'
    ]

    for frame_num in range(num_frames):
        # Simulate 3-6 objects per frame
        num_objects = random.randint(3, 6)
        objects = []

        for _ in range(num_objects):
            obj_class = random.choice(object_classes)

            # Random bbox
            x = random.randint(100, 1700)
            y = random.randint(100, 900)
            w = random.randint(50, 200)
            h = random.randint(50, 200)

            objects.append({
                'class': obj_class,
                'confidence': random.uniform(0.6, 0.95),
                'bbox': [x, y, w, h]
            })

        results.append({
            'frame': frame_num,
            'objects': objects,
            'mode': 'yolo+sam+clip'
        })

    return results


def test_template_generation():
    """Test template generation from procedure text"""
    print("\n" + "="*60)
    print("TEST 1: Template Generation")
    print("="*60)

    # Load procedure
    procedure_file = project_root / "videos" / "DNA_Extraction.txt"

    if not procedure_file.exists():
        print(f"ERROR: Procedure file not found: {procedure_file}")
        return False

    steps = load_procedure_from_file(str(procedure_file))
    print(f"Loaded {len(steps)} steps from procedure file")

    # Generate template
    generator = ProcedureTemplateGenerator()
    template = generator.generate_template(
        title="DNA Extraction",
        user_id="test_user",
        step_descriptions=steps
    )

    print(f"\nTemplate generated:")
    print(f"  - ID: {template['template_id']}")
    print(f"  - Title: {template['title']}")
    print(f"  - Steps: {len(template['steps'])}")
    print(f"  - Est. Duration: {template['metadata']['estimated_duration']}")

    # Show first 3 steps
    print("\nFirst 3 steps:")
    for i, step in enumerate(template['steps'][:3], 1):
        print(f"  {i}. {step['expected_action']}: {step['description'][:60]}...")
        if step['parameters']:
            params = list(step['parameters'].keys())
            print(f"      Parameters: {', '.join(params)}")

    # Save template
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    template_path = output_dir / "dna_extraction_template.json"
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"\nTemplate saved to: {template_path}")

    return template


def test_tep_processing(template: dict):
    """Test TEP with simulated video data"""
    print("\n" + "="*60)
    print("TEST 2: TEP Processing with Simulated Video")
    print("="*60)

    # Create executor
    executor = ProcedureExecutor(
        template=template,
        frame_width=1920,
        frame_height=1080
    )

    # Generate simulated video results
    print("\nGenerating simulated video results...")
    video_results = create_simulated_video_results(num_frames=300)
    print(f"Generated {len(video_results)} frames")

    # Process video
    print("\nProcessing through TEP...")
    events = executor.process_video_results(video_results, fps=30.0)

    print(f"\nDetected {len(events)} events")

    # Show event summary
    if events:
        print("\nEvent Summary:")
        matched = sum(1 for e in events if e.matched_expectation)
        print(f"  - Matched template: {matched}/{len(events)}")

        # Show first few events
        print("\nFirst 5 events:")
        for i, event in enumerate(events[:5], 1):
            status = "MATCHED" if event.matched_expectation else "MISMATCH"
            print(f"  {i}. [{status}] {event.action_type.value}")
            print(f"      Step {event.step_number}: {event.step_description[:40]}...")
            print(f"      Confidence: {event.confidence:.2f}")

    return executor


def test_execution_log_export(executor):
    """Test execution log export"""
    print("\n" + "="*60)
    print("TEST 3: Execution Log Export")
    print("="*60)

    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    log_path = output_dir / "execution_log.json"
    executor.export_execution_log(str(log_path))

    print(f"\nExecution log saved to: {log_path}")

    # Load and show summary
    with open(log_path, 'r') as f:
        log = json.load(f)

    print("\nExecution Summary:")
    print(f"  - Procedure: {log['procedure_title']}")
    print(f"  - Total Steps: {log['total_steps']}")
    print(f"  - Completed Steps: {log['completed_steps']}")
    print(f"  - Completion Rate: {log['completion_rate']:.1%}")
    print(f"\nStatistics:")
    print(f"  - Total Events: {log['statistics']['total_events']}")
    print(f"  - Match Rate: {log['statistics']['match_rate']:.1%}")
    print(f"  - Avg Confidence: {log['statistics']['average_confidence']:.2f}")


def test_imports():
    """Test that all modules can be imported"""
    print("\n" + "="*60)
    print("TEST 0: Import Verification")
    print("="*60)

    tests = []

    try:
        from src.tep.data_structures import ActionType, FrameData, TEPEvent
        tests.append(("Data structures", True))
    except ImportError as e:
        tests.append(("Data structures", False, str(e)))

    try:
        from src.procedure.template_generator import ProcedureTemplateGenerator
        tests.append(("Template generator", True))
    except ImportError as e:
        tests.append(("Template generator", False, str(e)))

    try:
        from src.tep.temporal_event_parser import TemporalEventParser
        tests.append(("TEP", True))
    except ImportError as e:
        tests.append(("TEP", False, str(e)))

    try:
        from src.integration.procedure_executor import ProcedureExecutor
        tests.append(("Procedure executor", True))
    except ImportError as e:
        tests.append(("Procedure executor", False, str(e)))

    all_passed = True
    for test in tests:
        if test[1]:
            print(f"  OK: {test[0]}")
        else:
            print(f"  FAIL: {test[0]} - {test[2]}")
            all_passed = False

    return all_passed


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TEP End-to-End Test Suite")
    print("="*60)

    # Test imports
    if not test_imports():
        print("\nERROR: Import tests failed. Please check module implementations.")
        sys.exit(1)

    # Test template generation
    template = test_template_generation()
    if not template:
        print("\nERROR: Template generation failed.")
        sys.exit(1)

    # Test TEP processing
    executor = test_tep_processing(template)

    # Test execution log export
    test_execution_log_export(executor)

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nNext steps:")
    print("1. Check output/dna_extraction_template.json")
    print("2. Check output/execution_log.json")
    print("3. Run with real video: python process_video.py --help")
    print("\nTo tune detection accuracy, adjust thresholds in:")
    print("  - src/tep/window_manager.py (stability_threshold)")
    print("  - src/tep/action_classifier.py (rule thresholds)")


if __name__ == "__main__":
    main()
