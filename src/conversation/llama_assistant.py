"""
Conversational Llama Assistant
Uses procedural + spatial context to answer questions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional

# Try to import bitsandbytes for 8-bit quantization (CUDA only)
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class LlamaConversationalAssistant:
    """
    Conversational assistant with procedure + spatial awareness
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the conversational assistant

        Args:
            model_path: Optional custom HuggingFace model path
        """
        self.model_path = model_path or "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model = None
        self.tokenizer = None
        self.conversation_history: List[Dict] = []

    def load(self):
        """Load Llama on-demand"""
        if self.model is not None:
            return  # Already loaded

        print("Loading Llama for conversation...")

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
            print("✓ Conversational Llama loaded with 8-bit quantization")
        else:
            # Fallback to float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            if not torch.cuda.is_available():
                print("✓ Conversational Llama loaded in float16 (CPU mode)")
            else:
                print("✓ Conversational Llama loaded in float16")

        self.model.eval()

    def answer_question(
        self,
        question: str,
        procedural_context: Dict,
        spatial_context: Dict
    ) -> str:
        """
        Answer question with full context

        Args:
            question: "How do I use the micropipette?"
            procedural_context: {
                "current_step": 3,
                "step_description": "Add 200µL buffer",
                "completed_steps": [1, 2],
                "template": {...}
            }
            spatial_context: {
                "objects_detected": ["micropipette", "tube_1"],
                "recent_actions": [...],
                "equipment_available": [...]
            }

        Returns:
            Answer string
        """
        self.load()  # Ensure loaded

        # Build context-aware prompt
        prompt = self._build_conversational_prompt(
            question,
            procedural_context,
            spatial_context
        )

        # Generate response
        response = self._generate(prompt)

        # Add to history
        self.conversation_history.append({
            "question": question,
            "answer": response
        })

        return response

    def _build_conversational_prompt(
        self,
        question: str,
        proc_ctx: Dict,
        spatial_ctx: Dict
    ) -> str:
        """Build context-rich prompt"""

        # Extract context values with defaults
        current_step = proc_ctx.get('current_step', 1)
        step_desc = proc_ctx.get('step_description', 'Unknown')
        completed_steps = proc_ctx.get('completed_steps', [])

        objects = spatial_ctx.get('objects_detected', [])
        equipment = spatial_ctx.get('equipment_available', [])
        recent_actions = spatial_ctx.get('recent_actions', [])

        # Format recent actions
        recent_actions_str = ""
        if recent_actions:
            action_list = []
            for action in recent_actions[-3:]:  # Last 3 actions
                if isinstance(action, dict):
                    action_list.append(f"- {action.get('action', 'unknown')} at {action.get('timestamp', '?')}")
                else:
                    action_list.append(f"- {action}")
            recent_actions_str = f"\nRecent actions:\n" + "\n".join(action_list)

        # Build system prompt
        system = f"""You are a laboratory assistant helping a researcher perform a procedure.

CURRENT SITUATION:
- Step {current_step}: {step_desc}
- Completed steps: {completed_steps if completed_steps else 'None yet'}
- Objects visible: {', '.join(objects) if objects else 'None detected'}
- Equipment available: {', '.join(equipment) if equipment else 'Standard lab equipment'}{recent_actions_str}

INSTRUCTIONS:
- Provide clear, practical guidance
- Be specific about what the user should do with the equipment they have
- Reference the current step when relevant
- If the question is about technique, explain step-by-step
- Consider safety when giving advice"""

        user = question

        # Llama 3 chat format
        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted

    def _generate(self, prompt: str) -> str:
        """Generate conversational response"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,  # Higher temp for conversation
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def unload(self):
        """Unload model to free VRAM"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("✓ Conversational Llama unloaded")

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


# Singleton instance for global access
_assistant: Optional[LlamaConversationalAssistant] = None


def get_assistant() -> LlamaConversationalAssistant:
    """Get or create singleton assistant instance"""
    global _assistant
    if _assistant is None:
        _assistant = LlamaConversationalAssistant()
    return _assistant
