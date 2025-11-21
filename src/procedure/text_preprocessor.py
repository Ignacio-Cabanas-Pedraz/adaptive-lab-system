"""
Stage 1: Text Preprocessing
Normalizes and tokenizes user input
"""

import re
from typing import Dict, List


class TextPreprocessor:
    """
    Normalizes and tokenizes user input
    """

    def __init__(self):
        self.abbreviations = {
            'min': 'minutes',
            'sec': 'seconds',
            'hr': 'hours',
            'hrs': 'hours',
            'rpm': 'revolutions per minute',
            'rcf': 'relative centrifugal force',
            'RT': 'room temperature',
            'O/N': 'overnight',
            'µL': 'microliters',
            'mL': 'milliliters',
            'µg': 'micrograms',
            'mg': 'milligrams',
        }

    def preprocess(self, text: str) -> Dict:
        """
        Preprocess a single step description

        Returns:
            {
                'original': str,
                'normalized': str,
                'tokens': List[str],
                'measurements': List[Dict]
            }
        """
        original = text.strip()
        normalized = self._normalize_text(original)
        tokens = self._tokenize(normalized)
        measurements = self._identify_measurements(original)

        return {
            'original': original,
            'normalized': normalized,
            'tokens': tokens,
            'measurements': measurements
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text while preserving measurements"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Standardize common variations
        text = text.replace('µL', 'µL')  # Normalize micro symbol
        text = text.replace('uL', 'µL')

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens"""
        return re.findall(r'\b\w+\b', text.lower())

    def _identify_measurements(self, text: str) -> List[Dict]:
        """
        Pre-identify all measurements in text
        Returns list of {value, unit, position}
        """
        measurements = []

        # Pattern: number + optional space + unit
        pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Zµ°]+)'

        for match in re.finditer(pattern, text):
            measurements.append({
                'value': match.group(1),
                'unit': match.group(2),
                'full_text': match.group(0),
                'position': match.span()
            })

        return measurements
