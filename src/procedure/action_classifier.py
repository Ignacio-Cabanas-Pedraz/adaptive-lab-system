"""
Stage 3: Action Classification
Classifies step descriptions into action types using keyword matching
"""

from typing import Dict, List, Optional
import re


ACTION_KEYWORDS = {
    'transfer': ['add', 'transfer', 'pipette', 'pour', 'dispense', 'aliquot', 'inject'],
    'mix': ['mix', 'vortex', 'shake', 'stir', 'agitate', 'invert', 'swirl'],
    'heat': ['heat', 'incubate', 'warm', 'boil', 'autoclave'],
    'cool': ['cool', 'chill', 'freeze', 'ice', 'refrigerate'],
    'centrifuge': ['centrifuge', 'spin', 'pellet'],
    'wait': ['wait', 'incubate', 'rest', 'stand', 'equilibrate', 'overnight'],
    'measure': ['measure', 'weigh', 'check', 'monitor', 'record', 'read'],
    'filter': ['filter', 'strain', 'separate'],
    'dissolve': ['dissolve', 'resuspend', 'reconstitute'],
    'wash': ['wash', 'rinse', 'clean'],
}


class ActionClassifier:
    """
    Classifies step descriptions into action types
    """

    def __init__(self):
        self.action_keywords = ACTION_KEYWORDS

        # Build reverse lookup
        self.keyword_to_action = {}
        for action_type, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword not in self.keyword_to_action:
                    self.keyword_to_action[keyword] = []
                self.keyword_to_action[keyword].append(action_type)

    def classify(self, text: str, extracted_params: Dict) -> Dict:
        """
        Classify action type from step description

        Returns:
            {
                'action_type': str,
                'confidence': float,
                'alternative': Optional[str],
                'reasoning': str
            }
        """
        text_lower = text.lower()

        # Find matching keywords
        matches = []
        for keyword in self.keyword_to_action.keys():
            if keyword in text_lower:
                for action_type in self.keyword_to_action[keyword]:
                    matches.append((action_type, keyword))

        if not matches:
            return self._infer_from_context(text, extracted_params)

        # Count votes
        vote_counts = {}
        for action_type, keyword in matches:
            vote_counts[action_type] = vote_counts.get(action_type, 0) + 1

        # Get top action
        sorted_actions = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        top_action = sorted_actions[0][0]
        top_votes = sorted_actions[0][1]

        # Check for ambiguity
        alternative = None
        if len(sorted_actions) > 1 and sorted_actions[1][1] == top_votes:
            alternative = sorted_actions[1][0]
            top_action = self._disambiguate(text, top_action, alternative, extracted_params)

        # Calculate confidence
        confidence = min(1.0, top_votes / 3.0)

        # Apply parameter-based adjustments
        adjusted = self._adjust_with_parameters(top_action, extracted_params)
        if adjusted != top_action:
            alternative = top_action
            top_action = adjusted

        return {
            'action_type': top_action,
            'confidence': confidence,
            'alternative': alternative,
            'reasoning': self._explain_classification(top_action, matches)
        }

    def _disambiguate(self, text: str, action1: str, action2: str, params: Dict) -> str:
        """Disambiguate between two equally-weighted actions"""
        # Incubate can be "heat" or "wait"
        if set([action1, action2]) == {'heat', 'wait'}:
            temp = params.get('temperature')
            if temp and temp['value'] not in ['RT', 'room temperature']:
                return 'heat'
            return 'wait'

        # Temperature keyword can mean "heat" or "cool"
        if set([action1, action2]) == {'heat', 'cool'}:
            temp = params.get('temperature')
            if temp and isinstance(temp['value'], (int, float)) and temp['value'] < 20:
                return 'cool'
            return 'heat'

        return action1

    def _adjust_with_parameters(self, action_type: str, params: Dict) -> str:
        """Adjust action type based on parameters"""
        if params.get('speed') and action_type not in ['centrifuge']:
            return 'centrifuge'

        if params.get('volume') and action_type == 'wait':
            return 'transfer'

        if params.get('count') and 'pipett' in action_type:
            return 'mix'

        return action_type

    def _infer_from_context(self, text: str, params: Dict) -> Dict:
        """Infer action when no keywords match"""
        if params.get('volume'):
            return {
                'action_type': 'transfer',
                'confidence': 0.6,
                'alternative': None,
                'reasoning': 'Inferred from volume parameter'
            }

        if params.get('speed'):
            return {
                'action_type': 'centrifuge',
                'confidence': 0.9,
                'alternative': None,
                'reasoning': 'Inferred from speed parameter'
            }

        return {
            'action_type': 'wait',
            'confidence': 0.3,
            'alternative': None,
            'reasoning': 'Default assumption'
        }

    def _explain_classification(self, action_type: str, matches: List) -> str:
        """Generate human-readable explanation"""
        keywords_found = [kw for at, kw in matches if at == action_type]
        if not keywords_found:
            return "Inferred from context"
        return f"Keywords found: {', '.join(keywords_found)}"
