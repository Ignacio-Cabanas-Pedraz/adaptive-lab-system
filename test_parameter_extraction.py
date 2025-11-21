#!/usr/bin/env python3
"""
Test script to demonstrate improved parameter extraction
"""

import sys
sys.path.append('.')

from src.procedure.parameter_extractor import ParameterExtractor

def test_extraction():
    extractor = ParameterExtractor()

    print("=" * 70)
    print("Testing Improved Parameter Extraction")
    print("=" * 70)

    # Test 1: Multiple durations with seconds/minutes confusion
    test_cases = [
        "Embed samples in ice for 10 minutes",
        "Incubate for 5 minutes at 25°C, then wait 30 seconds",
        "Heat at 95°C for 2 hours and 30 minutes",
        "Centrifuge at 10,000 rpm for 10s",
        "Vortex for 30s then spin down",
        "Mix gently for 5min at room temperature",
        "Add 50 µL of buffer and wait 2m",
        "Incubate 1 hour at 37°C with 100 µL reagent"
    ]

    for text in test_cases:
        print(f"\n📝 Text: {text}")
        print("-" * 70)

        # Extract all parameters
        results = extractor.extract_all(text)

        if results['duration']:
            print(f"  ⏱️  Duration: {results['duration']['value']} {results['duration']['unit']}")

        if results['temperature']:
            temp = results['temperature']
            print(f"  🌡️  Temperature: {temp['value']}°{temp['unit']}")

        if results['volume']:
            vol = results['volume']
            print(f"  💧 Volume: {vol['value']} {vol['unit']}")

        if results['speed']:
            speed = results['speed']
            print(f"  💨 Speed: {speed['value']} {speed['unit']}")

        if not any(results.values()):
            print("  ❌ No parameters extracted")

    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_extraction()
