"""
Procedure Template Generator with Llama Validation
Main orchestrator that combines all stages including LLM validation
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional

from .text_preprocessor import TextPreprocessor
from .parameter_extractor import ParameterExtractor
from .action_classifier import ActionClassifier


class ProcedureTemplateGenerator:
    """
    Generates procedure templates with optional Llama validation

    Workflow:
    1. Process steps with regex (Stages 1-3)
    2. Load Llama (if enabled)
    3. Validate each step
    4. Unload Llama
    5. Assemble template
    """

    def __init__(self, use_llm_validation: bool = False, llm_device: str = "cuda"):
        """
        Initialize the template generator

        Args:
            use_llm_validation: Whether to use Llama for validation
            llm_device: Device for LLM ("cuda" or "cpu")
        """
        # Stages 1-3 (rule-based)
        self.preprocessor = TextPreprocessor()
        self.parameter_extractor = ParameterExtractor()
        self.action_classifier = ActionClassifier()

        # Stage 4 (Llama validation)
        self.use_llm = use_llm_validation
        self.llm_device = llm_device
        self.validator = None
        self.safety_db = None

        # Lazy import safety database
        if self.use_llm:
            from .llm_validator import ChemicalSafetyDatabase
            self.safety_db = ChemicalSafetyDatabase()

    def generate_template(
        self,
        title: str,
        user_id: str,
        step_descriptions: List[str]
    ) -> Dict:
        """
        Generate complete template from step descriptions

        Args:
            title: Template title
            user_id: User who created template
            step_descriptions: List of step description strings

        Returns:
            Complete template dictionary
        """
        print(f"Generating template: {title}")
        print(f"Processing {len(step_descriptions)} steps...")

        # STAGE 1-3: Regex processing
        processed_steps = []
        for i, desc in enumerate(step_descriptions, start=1):
            step = self._process_step_regex(i, desc)
            processed_steps.append(step)

        print("✓ Regex processing complete")

        # STAGE 4: Llama validation (if enabled)
        if self.use_llm:
            processed_steps = self._validate_with_llama(processed_steps)

        # Assemble final template
        template = self._assemble_template(
            title=title,
            user_id=user_id,
            processed_steps=processed_steps
        )

        completeness = template['metadata']['completeness_score']
        print(f"✓ Template complete ({completeness:.0%} completeness)")

        return template

    def _process_step_regex(self, step_number: int, description: str) -> Dict:
        """Stages 1-3: Regex processing"""
        # Stage 1: Preprocess
        preprocessed = self.preprocessor.preprocess(description)

        # Stage 2: Extract parameters
        extracted_params = self.parameter_extractor.extract_all(
            preprocessed['original']
        )

        # Stage 3: Classify action
        classified_action = self.action_classifier.classify(
            preprocessed['original'],
            extracted_params
        )

        # Build step
        step = {
            'step_number': step_number,
            'description': description,
            'expected_action': classified_action['action_type'],
            'parameters': {k: v for k, v in extracted_params.items() if v is not None},
            'confidence': classified_action['confidence'],
            'alternative_action': classified_action.get('alternative'),
            'expected_objects': [],
            'chemicals': [],
            'equipment': [],
            'safety': [],
            'technique': 'standard',
            'llm_validated': False
        }

        return step

    def _validate_with_llama(self, processed_steps: List[Dict]) -> List[Dict]:
        """Stage 4: Llama validation"""
        from .llm_validator import LlamaParameterValidator

        print("Loading Llama for validation...")
        self.validator = LlamaParameterValidator(device=self.llm_device)

        for i, step in enumerate(processed_steps, 1):
            print(f"  Validating step {i}/{len(processed_steps)}...", end='\r')

            validated = self.validator.validate_and_enhance_step(
                original_text=step['description'],
                regex_results=step['parameters'],
                classified_action=step['expected_action']
            )

            # Apply corrections
            self._apply_validation(step, validated)

        print(f"✓ Validated {len(processed_steps)} steps          ")

        # Unload Llama to free VRAM
        self.validator.unload()
        self.validator = None
        print("✓ Llama unloaded")

        return processed_steps

    def _apply_validation(self, step: Dict, validated: Dict):
        """Apply Llama validation results to step"""
        # Fix parameters
        if 'validation' in validated:
            for param_name, validation in validated['validation'].items():
                if validation.get('error'):
                    # Regex was wrong, use corrected value
                    step['parameters'][param_name] = validation['corrected']

        # Add enhanced data
        step['chemicals'] = validated.get('chemicals', [])
        step['equipment'] = validated.get('equipment', [])
        step['technique'] = validated.get('technique', 'standard')

        # Combine LLM + database safety notes
        llm_safety = validated.get('safety_notes', [])
        db_safety = []
        if self.safety_db:
            for chem in step['chemicals']:
                db_safety.extend(self.safety_db.get_safety_notes(chem))

        step['safety'] = list(set(llm_safety + db_safety))
        step['llm_validated'] = True

    def _assemble_template(
        self,
        title: str,
        user_id: str,
        processed_steps: List[Dict]
    ) -> Dict:
        """Assemble final template"""
        template_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        template = {
            'template_id': template_id,
            'title': title,
            'version': '1.0',
            'created_by': user_id,
            'created_at': now,
            'modified_at': now,
            'metadata': {
                'step_count': len(processed_steps),
                'estimated_duration': self._estimate_duration(processed_steps),
                'completeness_score': self._calculate_completeness(processed_steps),
                'llm_validated': self.use_llm,
                'status': 'draft'
            },
            'steps': processed_steps,
            'post_procedure': self._generate_cleanup(processed_steps)
        }

        return template

    def _estimate_duration(self, steps: List[Dict]) -> str:
        """Estimate total procedure duration"""
        total_minutes = 0

        for step in steps:
            duration = step['parameters'].get('duration')
            if duration:
                if duration['unit'] == 'minutes':
                    total_minutes += duration['value']
                elif duration['unit'] == 'hours':
                    total_minutes += duration['value'] * 60
                elif duration['unit'] == 'seconds':
                    total_minutes += duration['value'] / 60

        # Add baseline for steps without duration
        steps_without_duration = sum(
            1 for s in steps if not s['parameters'].get('duration')
        )
        total_minutes += steps_without_duration * 2

        if total_minutes < 60:
            return f"{int(total_minutes)} minutes"
        else:
            hours = total_minutes / 60
            return f"{hours:.1f} hours"

    def _calculate_completeness(self, steps: List[Dict]) -> float:
        """Calculate template completeness score"""
        if not steps:
            return 0.0

        total_points = 0
        earned_points = 0

        for step in steps:
            # 6 criteria per step
            total_points += 6

            # Check each criterion
            if step.get('expected_action'):
                earned_points += 1
            if step.get('parameters'):
                earned_points += 1
            if step.get('chemicals'):
                earned_points += 1
            if step.get('equipment'):
                earned_points += 1
            if step.get('safety'):
                earned_points += 1
            if step.get('confidence', 0) > 0.7:
                earned_points += 1

        return earned_points / total_points if total_points > 0 else 0.0

    def _generate_cleanup(self, steps: List[Dict]) -> List[str]:
        """Generate post-procedure cleanup steps"""
        cleanup = []

        # Check if procedure used chemicals
        has_chemicals = any(s.get('chemicals') for s in steps)

        if has_chemicals:
            cleanup.append("Dispose of chemical waste in designated container")
            cleanup.append("Clean all glassware and equipment")

        # Check for specific equipment
        all_equipment = []
        for s in steps:
            all_equipment.extend(s.get('equipment', []))

        if any('pipette' in eq.lower() for eq in all_equipment):
            cleanup.append("Discard used pipette tips in appropriate waste")

        if any('centrifuge' in eq.lower() for eq in all_equipment):
            cleanup.append("Wipe centrifuge rotor if necessary")

        # Standard cleanup
        cleanup.append("Wipe down work surface")
        cleanup.append("Remove PPE and wash hands")

        return cleanup
