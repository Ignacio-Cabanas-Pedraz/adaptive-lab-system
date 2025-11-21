"""
Llama 3 8B Validator
Validates and enhances regex-extracted parameters
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import Dict, List

# Try to import bitsandbytes for 8-bit quantization (CUDA only)
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class LlamaParameterValidator:
    """
    Uses Llama 3 8B to validate and fix regex extraction results
    """

    def __init__(self, device: str = "cuda", model_path: str = None):
        """
        Initialize Llama 3 8B with 8-bit quantization

        Args:
            device: Device to load model on ("cuda" or "cpu")
            model_path: Optional custom HuggingFace model path
        """
        self.device = device
        self.model_path = model_path or "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load Llama 3 8B with optional 8-bit quantization"""
        print(f"Loading Llama 3 8B for parameter validation...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Use 8-bit quantization if available (CUDA only)
        if HAS_BITSANDBYTES and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✓ Llama loaded with 8-bit quantization (9GB VRAM)")
        else:
            # Fallback to float16 (requires more memory)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            if not torch.cuda.is_available():
                print("✓ Llama loaded in float16 (CPU mode - will be slow)")
            else:
                print("✓ Llama loaded in float16 (16GB VRAM)")

        self.model.eval()

    def validate_and_enhance_step(
        self,
        original_text: str,
        regex_results: Dict,
        classified_action: str
    ) -> Dict:
        """
        Validate regex results and enhance with additional info

        Args:
            original_text: "Incubate for 30 minutes at 37°C"
            regex_results: {
                "duration": {"value": 30, "unit": "seconds"},
                "temperature": {"value": 37, "unit": "°C"}
            }
            classified_action: "heat"

        Returns:
            {
                "validation": {...},  # Fixed parameters
                "chemicals": [...],
                "equipment": [...],
                "safety_notes": [...]
            }
        """
        # Build prompt
        prompt = self._build_validation_prompt(
            original_text,
            regex_results,
            classified_action
        )

        # Generate
        response = self._generate(prompt)

        # Parse JSON response
        result = self._parse_response(response)

        return result

    def _build_validation_prompt(
        self,
        text: str,
        regex_results: Dict,
        action: str
    ) -> str:
        """Build Llama 3 Instruct format prompt"""

        system = """You are a laboratory procedure validator. Your job is to:
1. Check if regex-extracted parameters match the original text
2. Fix any errors (especially unit mistakes)
3. Extract chemicals and equipment mentioned
4. Generate relevant safety notes

CRITICAL: Output ONLY valid JSON, nothing else."""

        user = f"""Original step: "{text}"
Action type: {action}

Regex extracted these parameters:
{json.dumps(regex_results, indent=2)}

Validate these results. Fix any errors. Extract chemicals, equipment, safety notes.

Output format (JSON only):
{{
  "validation": {{
    "parameter_name": {{
      "regex_value": {{}},
      "corrected": {{}},
      "error": "description of error or null"
    }}
  }},
  "chemicals": ["name1", "name2"],
  "equipment": ["item1", "item2"],
  "safety_notes": ["note1", "note2"],
  "technique": "standard|careful|precise"
}}"""

        # Llama 3 Instruct chat format
        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted

    def _generate(self, prompt: str) -> str:
        """Generate response from Llama"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temp for factual validation
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON response from Llama"""

        try:
            # Remove markdown if present
            if '```' in response:
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]

            # Find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            data = json.loads(response)
            return data

        except Exception as e:
            print(f"Warning: Failed to parse Llama response: {e}")
            return {
                "validation": {},
                "chemicals": [],
                "equipment": [],
                "safety_notes": [],
                "technique": "standard"
            }

    def unload(self):
        """Free VRAM by unloading the model"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("✓ Llama unloaded (9GB VRAM freed)")


class ChemicalSafetyDatabase:
    """Local database to augment LLM safety notes"""

    HAZARDS = {
        'hcl': ['Corrosive', 'Wear gloves and goggles'],
        'hydrochloric acid': ['Corrosive', 'Wear gloves and goggles'],
        'naoh': ['Corrosive', 'Causes severe burns'],
        'sodium hydroxide': ['Corrosive', 'Causes severe burns'],
        'ethanol': ['Flammable', 'Irritant'],
        'methanol': ['Toxic', 'Flammable', 'Avoid inhalation'],
        'phenol': ['Toxic', 'Corrosive', 'Severe burn hazard'],
        'chloroform': ['Toxic', 'Carcinogen', 'Use fume hood'],
        'formaldehyde': ['Toxic', 'Carcinogen', 'Use fume hood'],
        'acetone': ['Flammable', 'Irritant'],
        'trizol': ['Toxic', 'Corrosive', 'Use fume hood'],
        'sds': ['Irritant', 'Avoid eye contact'],
        'sodium dodecyl sulfate': ['Irritant', 'Avoid eye contact'],
        'edta': ['Irritant'],
        'proteinase k': ['Irritant', 'Avoid inhalation'],
        'rnase': ['Handle with care', 'Keep sterile'],
        'dnase': ['Handle with care', 'Keep sterile'],
        'isopropanol': ['Flammable', 'Irritant'],
        'bromophenol blue': ['Irritant'],
        'ethidium bromide': ['Mutagen', 'Carcinogen', 'Use gloves'],
        'sybr green': ['Irritant', 'Light sensitive'],
        'agarose': ['Handle with care when hot'],
        'tbe': ['Irritant'],
        'tae': ['Irritant'],
        'pbs': ['Generally safe'],
        'lysis buffer': ['May contain hazardous components', 'Check composition'],
        'binding buffer': ['Check composition'],
        'wash buffer': ['Check composition'],
        'elution buffer': ['Check composition'],
    }

    @staticmethod
    def get_safety_notes(chemical_name: str) -> List[str]:
        """Get safety notes from database"""
        name_lower = chemical_name.lower().strip()
        return ChemicalSafetyDatabase.HAZARDS.get(name_lower, [])

    @staticmethod
    def get_all_hazards_for_chemicals(chemicals: List[str]) -> List[str]:
        """Get combined safety notes for multiple chemicals"""
        all_notes = []
        for chem in chemicals:
            notes = ChemicalSafetyDatabase.get_safety_notes(chem)
            all_notes.extend(notes)
        return list(set(all_notes))
