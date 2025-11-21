#!/usr/bin/env python
"""
Test script for Llama 3 8B integration
Tests parameter validation, template generation, and conversational assistant
"""

import sys
import os
import json
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_parameter_validation():
    """Test 1: Parameter validation with Llama"""
    print("\n" + "="*60)
    print("TEST 1: Parameter Validation")
    print("="*60)

    from src.procedure.llm_validator import LlamaParameterValidator

    validator = LlamaParameterValidator()

    # Test case: minutes→seconds bug
    print("\nTest case: Validating duration extraction...")
    result = validator.validate_and_enhance_step(
        original_text="Incubate for 30 minutes at 37°C",
        regex_results={
            "duration": {"value": 30, "unit": "seconds"},  # Bug: should be minutes
            "temperature": {"value": 37, "unit": "°C"}
        },
        classified_action="heat"
    )

    print("\nValidation result:")
    print(json.dumps(result, indent=2))

    # Check if correction was made
    if 'validation' in result and 'duration' in result['validation']:
        corrected = result['validation']['duration'].get('corrected', {})
        if corrected.get('unit') == 'minutes':
            print("\n✓ PASS: Correctly identified unit should be 'minutes'")
        else:
            print("\n✗ FAIL: Did not correct the unit error")
    else:
        print("\n? INCONCLUSIVE: Could not verify correction")

    validator.unload()
    return result


def test_template_generation(use_llm=True):
    """Test 2: Template generation with/without Llama"""
    print("\n" + "="*60)
    print(f"TEST 2: Template Generation (LLM={'enabled' if use_llm else 'disabled'})")
    print("="*60)

    from src.procedure.template_generator import ProcedureTemplateGenerator

    # Sample procedure steps
    steps = [
        "Add 500µL lysis buffer to sample",
        "Incubate for 30 minutes at room temperature",
        "Centrifuge at 10,000 rpm for 5 minutes",
        "Transfer supernatant to new tube",
        "Add 500µL ethanol and mix by pipetting"
    ]

    print(f"\nGenerating template with {len(steps)} steps...")

    generator = ProcedureTemplateGenerator(use_llm_validation=use_llm)
    template = generator.generate_template(
        title="DNA Extraction Test",
        user_id="test_user",
        step_descriptions=steps
    )

    print("\nTemplate metadata:")
    print(f"  - Template ID: {template['template_id'][:8]}...")
    print(f"  - Steps: {template['metadata']['step_count']}")
    print(f"  - Duration: {template['metadata']['estimated_duration']}")
    print(f"  - Completeness: {template['metadata']['completeness_score']:.0%}")
    print(f"  - LLM Validated: {template['metadata']['llm_validated']}")

    # Show sample step
    if template['steps']:
        step = template['steps'][0]
        print(f"\nSample step (Step 1):")
        print(f"  - Description: {step['description']}")
        print(f"  - Action: {step['expected_action']}")
        print(f"  - Chemicals: {step.get('chemicals', [])}")
        print(f"  - Equipment: {step.get('equipment', [])}")
        print(f"  - Safety: {step.get('safety', [])}")

    # Save template
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_template.json")

    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"\n✓ Template saved to: {output_path}")

    return template


def test_conversational_assistant():
    """Test 3: Conversational assistant"""
    print("\n" + "="*60)
    print("TEST 3: Conversational Assistant")
    print("="*60)

    from src.conversation.llama_assistant import get_assistant

    assistant = get_assistant()

    # Test context
    procedural_context = {
        "current_step": 3,
        "step_description": "Add 200µL buffer to sample",
        "completed_steps": [1, 2],
        "upcoming_steps": [4, 5, 6]
    }

    spatial_context = {
        "objects_detected": ["micropipette", "tube_1", "tube_2", "beaker"],
        "recent_actions": [
            {"action": "transfer", "timestamp": "45s"},
            {"action": "mix", "timestamp": "52s"}
        ],
        "equipment_available": ["P200 pipette", "P1000 pipette", "vortex", "incubator"]
    }

    # Test questions
    questions = [
        "How do I use the micropipette for this step?",
        "What should I do if I accidentally add too much buffer?"
    ]

    for question in questions:
        print(f"\nQ: {question}")

        answer = assistant.answer_question(
            question=question,
            procedural_context=procedural_context,
            spatial_context=spatial_context
        )

        print(f"A: {answer}")
        print("-" * 40)

    # Show history
    history = assistant.get_history()
    print(f"\n✓ Conversation history: {len(history)} exchanges")

    assistant.unload()
    return history


def test_full_pipeline():
    """Test 4: Full pipeline with video context"""
    print("\n" + "="*60)
    print("TEST 4: Full Pipeline Simulation")
    print("="*60)

    # Check if DNA_Extraction.txt exists
    procedure_file = "videos/DNA_Extraction.txt"

    if os.path.exists(procedure_file):
        print(f"\nFound procedure file: {procedure_file}")

        with open(procedure_file) as f:
            content = f.read()
            steps = [line.strip() for line in content.split('\n') if line.strip()]

        print(f"Loaded {len(steps)} steps from file")

        from src.procedure.template_generator import ProcedureTemplateGenerator

        generator = ProcedureTemplateGenerator(use_llm_validation=True)
        template = generator.generate_template(
            title="DNA Extraction (from file)",
            user_id="test_user",
            step_descriptions=steps
        )

        print(f"\n✓ Full pipeline complete")
        print(f"  - Completeness: {template['metadata']['completeness_score']:.0%}")

        return template
    else:
        print(f"\n⚠ Procedure file not found: {procedure_file}")
        print("  Skipping full pipeline test")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Llama 3 8B integration")
    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4],
                       help='Run specific test (1-4)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Run template generation without LLM')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')

    args = parser.parse_args()

    print("="*60)
    print("Llama 3 8B Integration Test Suite")
    print("="*60)

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA: Available ({device_name})")
        else:
            print("CUDA: Not available (will use CPU)")
    except ImportError:
        print("PyTorch not installed")
        sys.exit(1)

    # Run tests
    if args.all:
        test_parameter_validation()
        test_template_generation(use_llm=not args.no_llm)
        test_conversational_assistant()
        test_full_pipeline()
    elif args.test == 1:
        test_parameter_validation()
    elif args.test == 2:
        test_template_generation(use_llm=not args.no_llm)
    elif args.test == 3:
        test_conversational_assistant()
    elif args.test == 4:
        test_full_pipeline()
    else:
        # Default: run test 2 without LLM (quick test)
        print("\nRunning quick test (template generation without LLM)...")
        print("Use --all to run full test suite with LLM")
        test_template_generation(use_llm=False)

    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
