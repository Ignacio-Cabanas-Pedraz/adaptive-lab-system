"""
Stage 2: Parameter Extraction
Extracts volumes, temperatures, durations, speeds using regex
"""

import re
from typing import Dict, List, Optional


class ParameterExtractor:
    """
    Extracts parameters using regex patterns
    """

    def __init__(self):
        self.patterns = {
            'volume': self._compile_volume_patterns(),
            'temperature': self._compile_temperature_patterns(),
            'duration': self._compile_duration_patterns(),
            'speed': self._compile_speed_patterns(),
            'count': self._compile_count_patterns(),
            'concentration': self._compile_concentration_patterns()
        }

    def extract_all(self, text: str) -> Dict:
        """Extract all parameters from text"""
        return {
            'volume': self.extract_volume(text),
            'temperature': self.extract_temperature(text),
            'duration': self.extract_duration(text),
            'speed': self.extract_speed(text),
            'count': self.extract_count(text),
            'concentration': self.extract_concentration(text)
        }

    def _compile_volume_patterns(self) -> List[re.Pattern]:
        """Patterns for volumes"""
        return [
            re.compile(r'(\d+(?:\.\d+)?)\s*([µu]L|microliters?)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(mL|milliliters?)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(L|liters?)', re.IGNORECASE),
        ]

    def extract_volume(self, text: str) -> Optional[Dict]:
        """Extract volume with normalized unit, preferring the last/most relevant occurrence"""
        all_matches = []

        for pattern in self.patterns['volume']:
            for match in pattern.finditer(text):
                value = float(match.group(1))
                unit_raw = match.group(2)
                unit_normalized = self._normalize_volume_unit(unit_raw)

                all_matches.append({
                    'value': value,
                    'unit': unit_normalized,
                    'raw': match.group(0),
                    'position': match.start()
                })

        if not all_matches:
            return None

        # Return the last occurrence (often most relevant in procedural text)
        return all_matches[-1]

    def _normalize_volume_unit(self, unit: str) -> str:
        """Normalize volume units"""
        unit_lower = unit.lower()
        if 'µl' in unit_lower or 'ul' in unit_lower or 'microliter' in unit_lower:
            return 'µL'
        elif 'ml' in unit_lower or 'milliliter' in unit_lower:
            return 'mL'
        elif unit_lower in ['l', 'liter', 'liters']:
            return 'L'
        return unit

    def _compile_temperature_patterns(self) -> List[re.Pattern]:
        return [
            re.compile(r'(-?\d+(?:\.\d+)?)\s*°?\s*C(?![a-z])', re.IGNORECASE),
            re.compile(r'room\s+temp(?:erature)?', re.IGNORECASE),
        ]

    def extract_temperature(self, text: str) -> Optional[Dict]:
        """Extract temperature, handling multiple values"""
        # Check for room temperature first
        if re.search(r'room\s+temp(?:erature)?|\bRT\b', text, re.IGNORECASE):
            return {'value': 'RT', 'unit': '°C', 'raw': 'room temperature'}

        all_matches = []

        for pattern in self.patterns['temperature']:
            for match in pattern.finditer(text):
                try:
                    value = float(match.group(1))
                    all_matches.append({
                        'value': value,
                        'unit': '°C',
                        'raw': match.group(0),
                        'position': match.start()
                    })
                except (ValueError, IndexError):
                    continue

        if not all_matches:
            return None

        # Return the last occurrence
        return all_matches[-1]

    def _compile_duration_patterns(self) -> List[re.Pattern]:
        """
        Ordered from most specific to least specific to avoid false matches.
        Uses word boundaries and negative lookaheads to prevent matching parts of words.
        """
        return [
            # Hours (most specific)
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b', re.IGNORECASE),
            # Minutes (medium specific - must have 'min' or standalone 'm')
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*m\b(?!inutes?)', re.IGNORECASE),  # Standalone 'm' but not part of 'minutes'
            # Seconds (specific - must have 'sec' or standalone 's')
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*s\b(?!ec)', re.IGNORECASE),  # Standalone 's' but not part of 'seconds'
        ]

    def extract_duration(self, text: str) -> Optional[Dict]:
        """Extract duration, preferring the most specific/explicit match"""
        all_matches = []

        # Collect all matches with their patterns
        for i, pattern in enumerate(self.patterns['duration']):
            for match in pattern.finditer(text):
                value = float(match.group(1))
                match_text = match.group(0).lower()

                # Determine unit based on pattern index and match content
                if i == 0 or 'hour' in match_text or 'hr' in match_text:
                    unit = 'hours'
                elif i <= 2 and ('minute' in match_text or 'min' in match_text):
                    unit = 'minutes'
                elif i == 2 and match_text.strip().endswith('m'):
                    unit = 'minutes'
                elif 'second' in match_text or 'sec' in match_text:
                    unit = 'seconds'
                elif match_text.strip().endswith('s'):
                    unit = 'seconds'
                else:
                    unit = 'minutes'  # Default fallback

                all_matches.append({
                    'value': value,
                    'unit': unit,
                    'raw': match.group(0),
                    'position': match.start(),
                    'specificity': i  # Lower index = more specific
                })

        if not all_matches:
            return None

        # Prefer matches with higher specificity (lower index), then last occurrence
        all_matches.sort(key=lambda x: (x['specificity'], -x['position']))
        best_match = all_matches[0]

        return {
            'value': best_match['value'],
            'unit': best_match['unit'],
            'raw': best_match['raw']
        }

    def _compile_speed_patterns(self) -> List[re.Pattern]:
        return [
            re.compile(r'(\d+(?:,\d+)?)\s*rpm', re.IGNORECASE),
            re.compile(r'(\d+(?:,\d+)?)\s*(?:x\s*)?g(?![a-z])', re.IGNORECASE),
        ]

    def extract_speed(self, text: str) -> Optional[Dict]:
        """Extract speed (RPM or g-force), handling multiple values"""
        all_matches = []

        for pattern in self.patterns['speed']:
            for match in pattern.finditer(text):
                value_str = match.group(1).replace(',', '')
                value = float(value_str)
                unit = 'rpm' if 'rpm' in match.group(0).lower() else 'g'
                all_matches.append({
                    'value': value,
                    'unit': unit,
                    'raw': match.group(0),
                    'position': match.start()
                })

        if not all_matches:
            return None

        # Return the last occurrence
        return all_matches[-1]

    def _compile_count_patterns(self) -> List[re.Pattern]:
        return [
            re.compile(r'(\d+)\s*(?:times|x)', re.IGNORECASE),
            re.compile(r'\b(once|twice|thrice)\b', re.IGNORECASE),
        ]

    def extract_count(self, text: str) -> Optional[Dict]:
        match = self.patterns['count'][0].search(text)
        if match:
            return {'value': int(match.group(1)), 'unit': 'repetitions', 'raw': match.group(0)}

        match = self.patterns['count'][1].search(text)
        if match:
            word = match.group(1).lower()
            count_map = {'once': 1, 'twice': 2, 'thrice': 3}
            return {'value': count_map[word], 'unit': 'repetitions', 'raw': word}
        return None

    def _compile_concentration_patterns(self) -> List[re.Pattern]:
        return [
            re.compile(r'(\d+(?:\.\d+)?)\s*([µmu]?M)(?![a-z])', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*%', re.IGNORECASE),
        ]

    def extract_concentration(self, text: str) -> Optional[Dict]:
        """Extract concentration, handling multiple values"""
        all_matches = []

        for pattern in self.patterns['concentration']:
            for match in pattern.finditer(text):
                value = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else '%'
                all_matches.append({
                    'value': value,
                    'unit': unit,
                    'raw': match.group(0),
                    'position': match.start()
                })

        if not all_matches:
            return None

        # Return the last occurrence
        return all_matches[-1]
