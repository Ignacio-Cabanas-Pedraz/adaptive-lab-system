# Semantic Binding System Implementation Guide
## Object-to-Concept Mapping for Laboratory Procedures

---

## Overview

This document describes the implementation of the **Semantic Binding Agent** - a system that maps physical objects (what the camera sees) to semantic concepts (what the procedure refers to). This enables precise, meaningful procedure tracking where "test_tube_1" becomes "compound A stock solution" and the AI can reason about chemical relationships, not just physical movements.

**Purpose**: Enable the AI to maintain chemically and procedurally accurate logs like:
- ✅ "Added 200µL compound A (test_tube_1) to solution B (petri_dish_1)"
- ❌ "Added liquid from test_tube_1 to petri_dish_1" (too vague)

**Integration**: Works alongside the existing vision pipeline (README.md system) as an interception layer between vision output and procedure logging.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Integration Points](#integration-points)
4. [Model Selection & Deployment](#model-selection--deployment)
5. [Implementation Details](#implementation-details)
6. [Data Structures](#data-structures)
7. [Workflow Examples](#workflow-examples)
8. [Performance Optimization](#performance-optimization)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)
11. [Configuration](#configuration)
12. [Deployment Guide](#deployment-guide)

---

## Architecture Overview

### High-Level Design

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM ARCHITECTURE                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Smart Glasses Camera                                            │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LAYER 1: VISION ENSEMBLE (Existing System)            │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │     │
│  │  │  YOLO    │→ │  SAM 2   │→ │  CLIP    │            │     │
│  │  └──────────┘  └──────────┘  └──────────┘            │     │
│  │                                                        │     │
│  │  Output: Physical Description                         │     │
│  │  {                                                     │     │
│  │    objects: [                                          │     │
│  │      {id: "test_tube_1", class: "test tube",          │     │
│  │       description: "test tube with yellow liquid"},   │     │
│  │      {id: "petri_dish_1", class: "petri dish",        │     │
│  │       description: "petri dish with clear solution"}  │     │
│  │    ],                                                  │     │
│  │    scene: "Lab bench with test tube on left..."       │     │
│  │  }                                                     │     │
│  └────────────────────────────────────────────────────────┘     │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LAYER 2: ACTION DETECTION (NEW)                       │     │
│  │                                                         │     │
│  │  Compares frame N-1 to frame N:                        │     │
│  │  - Object position changes?                            │     │
│  │  - Pipette movement detected?                          │     │
│  │  - Transfer action occurred?                           │     │
│  │                                                         │     │
│  │  Output: Action or None                                │     │
│  │  {                                                      │     │
│  │    type: "transfer",                                   │     │
│  │    source: "test_tube_1",                              │     │
│  │    destination: "petri_dish_1",                        │     │
│  │    tool: "pipette",                                    │     │
│  │    timestamp: "10:08:15"                               │     │
│  │  }                                                      │     │
│  └────────────────────────────────────────────────────────┘     │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LAYER 3: SEMANTIC BINDING AGENT (NEW - CORE)          │     │
│  │                                                         │     │
│  │  Lightweight LLM (Llama 3.2 3B):                       │     │
│  │                                                         │     │
│  │  Receives:                                              │     │
│  │  - Physical description (from Layer 1)                 │     │
│  │  - Action detected (from Layer 2)                      │     │
│  │  - Current procedure step text                         │     │
│  │  - Existing binding table                              │     │
│  │                                                         │     │
│  │  Reasons:                                               │     │
│  │  "Procedure says 'add compound A'. User drew from      │     │
│  │   test_tube_1. Therefore: test_tube_1 = compound A"    │     │
│  │                                                         │     │
│  │  Output: Enhanced Description                          │     │
│  │  {                                                      │     │
│  │    objects: [                                           │     │
│  │      {id: "test_tube_1",                               │     │
│  │       semantic_label: "compound A stock solution",     │     │
│  │       confidence: 0.92},                               │     │
│  │      {id: "petri_dish_1",                              │     │
│  │       semantic_label: "solution B with compound A",    │     │
│  │       confidence: 0.95}                                │     │
│  │    ],                                                   │     │
│  │    semantic_scene: "Compound A stock (test_tube_1)     │     │
│  │                     and solution B (petri_dish_1)..."  │     │
│  │  }                                                      │     │
│  └────────────────────────────────────────────────────────┘     │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LAYER 4: PROCEDURE CONTEXT (Existing System)          │     │
│  │                                                         │     │
│  │  Receives enhanced description with semantic labels    │     │
│  │                                                         │     │
│  │  Procedure Log Entry:                                  │     │
│  │  [10:08, lab_bench_a] Transferred 200µL compound A     │     │
│  │                       (test_tube_1) to solution B      │     │
│  │                       (petri_dish_1). No color change. │     │
│  │                                                         │     │
│  │  Binding Table Updated:                                │     │
│  │  test_tube_1 → compound A stock (confidence: 0.92)     │     │
│  │  petri_dish_1 → solution B + compound A (conf: 0.95)   │     │
│  └────────────────────────────────────────────────────────┘     │
│         ↓                                                        │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  LAYER 5: MAIN AI REASONING (Existing System)          │     │
│  │                                                         │     │
│  │  Llama 3 70B receives:                                 │     │
│  │  - Full procedure log (with semantic labels)           │     │
│  │  - Binding table                                       │     │
│  │  - Current workspace context                           │     │
│  │                                                         │     │
│  │  Generates contextual guidance:                        │     │
│  │  "Compound A has been added to solution B. The yellow  │     │
│  │   color in test_tube_1 is characteristic of compound A.│     │
│  │   Solution B should turn yellow within 2 minutes if    │     │
│  │   reaction is proceeding correctly. Monitor for color  │     │
│  │   change. Step 5/8."                                   │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Innovation

**Before Semantic Binding:**
```
Vision: "Liquid transferred from test_tube_1 to petri_dish_1"
Log: "Transferred liquid between containers"
AI: "You moved something somewhere"
```

**After Semantic Binding:**
```
Vision: "Liquid transferred from test_tube_1 to petri_dish_1"
Binding: test_tube_1 = compound A, petri_dish_1 = solution B
Log: "Transferred 200µL compound A to solution B"
AI: "Compound A is now in solution B. The reaction mixture contains
     both reactants. Monitor for expected yellow color change."
```

### Design Principles

1. **Lightweight**: Uses small model (3B) for fast inference (50-200ms)
2. **Conditional**: Only runs when actions detected (~10-20% of frames)
3. **Probabilistic**: Maintains confidence scores for bindings
4. **Self-correcting**: Updates bindings based on new observations
5. **Transparent**: Logs reasoning for debugging

---

## System Components

### 1. Action Detector

**Purpose**: Detect when user performs actions that require semantic binding

**Input**: Current frame objects + previous frame objects

**Output**: Action description or None

**Detection Methods**:
- **Pipette Movement**: Detects significant position change of pipette
- **Object Proximity**: Tracks which objects pipette approaches
- **Sequence Recognition**: Identifies pickup → move → dispense pattern

**Example**:
```python
Previous frame: pipette at (100, 200) near test_tube_1
Current frame:  pipette at (450, 300) near petri_dish_1

Action detected:
{
    'type': 'transfer',
    'source': 'test_tube_1',
    'destination': 'petri_dish_1',
    'tool': 'pipette',
    'confidence': 0.85
}
```

### 2. Semantic Binding Agent

**Purpose**: Map physical objects to semantic concepts using LLM reasoning

**Model**: Llama 3.2 3B Instruct (FP16, ~7GB VRAM)

**Input**:
- Vision description (physical objects)
- Detected action
- Current procedure step text
- Existing binding table

**Output**:
- Updated binding table
- Enhanced object descriptions with semantic labels
- Confidence scores

**Reasoning Process**:
1. Parse procedure step for semantic entities ("compound A", "solution B")
2. Observe which physical objects were involved in action
3. Match semantic entities to physical objects based on:
   - Temporal alignment (step mentions A, user used object X)
   - Spatial context (left tube, right dish)
   - Visual features (yellow liquid matches compound A description)
   - Prior bindings (consistency with previous assignments)
4. Generate binding with confidence score
5. Update binding table

### 3. Binding Table Manager

**Purpose**: Persistent storage and retrieval of object-semantic mappings

**Storage**: JSON file per procedure session

**Operations**:
- `add_binding(object_id, semantic_label, confidence)`
- `get_binding(object_id) → semantic_label`
- `update_binding(object_id, new_label, new_confidence)`
- `list_bindings() → all current bindings`
- `clear_deprecated_bindings()` - remove old/uncertain bindings

**Data Structure**:
```json
{
    "procedure_id": "protein_purification_20250316",
    "bindings": {
        "test_tube_1": {
            "semantic_label": "compound A stock solution",
            "confidence": 0.92,
            "assigned_at": "step_3",
            "first_seen": "2025-03-16T10:05:23Z",
            "last_updated": "2025-03-16T10:08:15Z",
            "status": "active",
            "physical_description": "test tube with yellow liquid",
            "location": "lab_bench_a"
        },
        "petri_dish_1": {
            "semantic_label": "solution B with compound A added",
            "confidence": 0.95,
            "assigned_at": "step_4",
            "first_seen": "2025-03-16T10:03:10Z",
            "last_updated": "2025-03-16T10:08:15Z",
            "status": "active",
            "contains": ["solution_B_original", "compound_A_200uL"],
            "location": "lab_bench_a"
        },
        "beaker_1": {
            "semantic_label": "50mM Tris buffer",
            "confidence": 0.98,
            "assigned_at": "step_1",
            "first_seen": "2025-03-16T09:45:10Z",
            "last_updated": "2025-03-16T09:50:30Z",
            "status": "depleted",
            "location": "prep_bench"
        }
    }
}
```

### 4. Enhanced Procedure Logger

**Purpose**: Record actions with both physical and semantic information

**Input**: Enhanced vision output (with semantic labels)

**Output**: Procedure log entries with dual representation

**Format**:
```json
{
    "step": 4,
    "timestamp": "2025-03-16T10:08:15Z",
    "workspace": "lab_bench_a",
    "action": {
        "type": "transfer",
        "physical": {
            "source": "test_tube_1",
            "destination": "petri_dish_1",
            "tool": "pipette_p200",
            "volume_visual": "~200µL"
        },
        "semantic": {
            "source_label": "compound A stock solution",
            "destination_label": "solution B",
            "purpose": "add compound A to solution B per protocol step 4"
        },
        "binding_confidence": 0.93
    },
    "observation": {
        "color_change": "none",
        "temperature": "23°C",
        "notes": "No visible reaction yet"
    },
    "result": {
        "physical": "petri_dish_1 now contains mixture",
        "semantic": "solution B now contains compound A (200µL)"
    }
}
```

---

## Integration Points

### Integration with Existing System

The semantic binding system integrates at **one specific point** in the existing pipeline:

**Insertion Point**: Between vision output and procedure logging

```python
# EXISTING PIPELINE (from README.md)

def process_frame_OLD(self, frame, imu_data):
    # Vision processing
    vision_output = self.vision_pipeline.process(frame)
    
    # Workspace memory
    workspace_result = self.workspace_memory.process(vision_output, imu_data)
    
    # Procedure logging
    self.procedure_log.append(vision_output)  # ← PHYSICAL ONLY
    
    # AI guidance
    guidance = self.llm.generate(procedure_log, workspace_result)
    
    return guidance


# NEW PIPELINE (with semantic binding)

def process_frame_NEW(self, frame, imu_data):
    # Vision processing (unchanged)
    vision_output = self.vision_pipeline.process(frame)
    
    # ┌─────────────────────────────────────────────┐
    # │  NEW: SEMANTIC BINDING INTERCEPTION         │
    # └─────────────────────────────────────────────┘
    
    # Detect action
    action = self.action_detector.detect(vision_output)
    
    # Semantic binding (only if action detected)
    if action:
        enhanced_output = self.semantic_binding.process(
            vision_output,
            action,
            self.current_procedure_step
        )
    else:
        # No action - just add existing semantic labels
        enhanced_output = self.semantic_binding.add_existing_labels(
            vision_output
        )
    
    # ┌─────────────────────────────────────────────┐
    # │  EXISTING PIPELINE CONTINUES (unchanged)    │
    # └─────────────────────────────────────────────┘
    
    # Workspace memory receives enhanced output
    workspace_result = self.workspace_memory.process(enhanced_output, imu_data)
    
    # Procedure logging receives enhanced output (now has semantic labels)
    self.procedure_log.append(enhanced_output)  # ← PHYSICAL + SEMANTIC
    
    # AI guidance receives full context
    guidance = self.llm.generate(procedure_log, workspace_result)
    
    return guidance
```

### File Modifications Required

**Minimal changes to existing codebase:**

1. **`lab_vision_system.py`** (main system class)
   - Add: `self.semantic_binding = SemanticBindingAgent()`
   - Add: `self.action_detector = ActionDetector()`
   - Modify: `process_frame()` to include semantic binding step

2. **`core/memory/procedure_context.py`**
   - Modify: Accept enhanced output with semantic labels
   - Add: Store binding table alongside procedure log

3. **`config/memory_config.yaml`**
   - Add: Semantic binding configuration section

**New files to create:**

```
core/reasoning/
├── semantic_binding_agent.py      # NEW: Main binding logic
├── action_detector.py              # NEW: Action detection
├── binding_table.py                # NEW: Binding storage
└── prompt_templates.py             # NEW: LLM prompts
```

---

## Model Selection & Deployment

### Recommended Model: Llama 3.2 3B Instruct

**Why this model:**

| Criterion | Llama 3.2 3B | Alternatives |
|-----------|--------------|--------------|
| **Size** | 3B params | Phi-3 Mini (3.8B), Mistral 7B |
| **VRAM** | 6-8GB FP16 | 7-9GB, 14-16GB |
| **Speed** | 50-200ms | 60-150ms, 100-300ms |
| **Quality** | Good for entity binding | Similar, Better |
| **JSON output** | Excellent | Good, Excellent |
| **Cost** | Free (open source) | Free, Free |

**Performance on RTX 4090:**
- Inference: 80-150ms typical
- Tokenization: 5-10ms
- Total: 85-160ms per action

### Memory Budget Analysis

**Single RunPod RTX 4090 Instance (24GB VRAM):**

```
┌──────────────────────────────────────────────────┐
│           VRAM ALLOCATION                        │
├──────────────────────────────────────────────────┤
│                                                  │
│  Vision Pipeline:                                │
│  ├─ YOLO v8 nano:              0.5 GB            │
│  ├─ SAM 2 tiny:                2.5 GB            │
│  ├─ CLIP ViT-B/32:             2.0 GB            │
│  └─ Subtotal:                  5.0 GB            │
│                                                  │
│  Semantic Binding:                               │
│  ├─ Llama 3.2 3B (FP16):       7.0 GB            │
│  └─ Subtotal:                  7.0 GB            │
│                                                  │
│  Memory & Buffers:                               │
│  ├─ Workspace embeddings:      1.0 GB            │
│  ├─ Frame buffers:             0.5 GB            │
│  ├─ Binding table cache:       0.2 GB            │
│  └─ Subtotal:                  1.7 GB            │
│                                                  │
│  Main AI (optional on same GPU):                │
│  └─ Llama 3 8B (8-bit):       10.0 GB            │
│                                                  │
├──────────────────────────────────────────────────┤
│  TOTAL (without main AI):     13.7 GB / 24 GB    │
│  TOTAL (with main AI):        23.7 GB / 24 GB    │
│  Headroom:                    10.3 GB / 0.3 GB   │
└──────────────────────────────────────────────────┘

✅ Fits comfortably without main AI
⚠️ Tight fit with main AI on same GPU
```

**Recommendation**: 
- **Option A**: Run main AI (Llama 3) via API (OpenAI/Anthropic) - semantic binding on GPU
- **Option B**: Use two GPUs (one for vision+binding, one for main AI)
- **Option C**: Use 8-bit quantization for all models (saves 50% VRAM)

### Alternative Configurations

**If VRAM constrained:**

```python
# Option 1: Smaller model
model = "meta-llama/Llama-3.2-1B-Instruct"  # Only 2-3GB VRAM
# Trade-off: Slightly lower binding accuracy (85% vs 90%)

# Option 2: 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    load_in_8bit=True,  # Reduces VRAM by 50%
    device_map="auto"
)
# Trade-off: 10-20ms slower inference

# Option 3: CPU inference (last resort)
device = "cpu"  # Frees GPU VRAM entirely
# Trade-off: 500-1000ms inference (very slow)
```

### Installation

**On RunPod instance:**

```bash
# Install transformers and dependencies
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3  # For 8-bit quantization
pip install sentencepiece==0.1.99  # For tokenization

# Download model (one-time, ~6GB)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='float16',
    device_map='auto'
)
print('Model downloaded successfully')
"
```

---

## Implementation Details

### 1. SemanticBindingAgent Class

**Location**: `core/reasoning/semantic_binding_agent.py`

**Full Implementation**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from typing import Dict, List, Optional

class SemanticBindingAgent:
    """
    Maps physical objects to semantic concepts using LLM reasoning.
    
    This agent intercepts vision output and enhances it with semantic
    labels by reasoning about procedure context and observed actions.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "auto",
        quantization: Optional[str] = None  # "8bit" or "4bit"
    ):
        """
        Initialize semantic binding agent.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cuda", "cpu", or "auto"
            quantization: None, "8bit", or "4bit"
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading semantic binding model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optional quantization
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device
        }
        
        if quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Binding table (persistent across frames)
        self.binding_table = {}
        
        # Performance tracking
        self.inference_times = []
        
        print("✓ Semantic binding agent initialized")
    
    def process(
        self,
        vision_output: Dict,
        action_detected: Optional[Dict],
        procedure_step: str,
        procedure_history: Optional[List] = None
    ) -> Dict:
        """
        Main processing: enhance vision output with semantic labels.
        
        Args:
            vision_output: Output from vision pipeline
            action_detected: Detected action or None
            procedure_step: Current procedure step text
            procedure_history: Recent procedure log entries
            
        Returns:
            Enhanced vision output with semantic labels
        """
        start_time = time.time()
        
        # If no action, just add existing bindings (fast path)
        if action_detected is None:
            enhanced = self._add_existing_bindings(vision_output)
            return enhanced
        
        # Build prompt for LLM
        prompt = self._build_prompt(
            vision_output,
            action_detected,
            procedure_step,
            procedure_history or []
        )
        
        # Get LLM inference
        bindings_result = self._infer_bindings(prompt)
        
        # Update binding table
        self._update_binding_table(bindings_result)
        
        # Enhance vision output
        enhanced = self._enhance_output(vision_output, bindings_result)
        
        # Track performance
        elapsed = time.time() - start_time
        self.inference_times.append(elapsed)
        
        return enhanced
    
    def _build_prompt(
        self,
        vision_output: Dict,
        action: Dict,
        step: str,
        history: List
    ) -> str:
        """
        Build LLM prompt for semantic binding inference.
        
        Prompt engineering notes:
        - Keep concise for speed (fewer tokens = faster)
        - Request structured JSON output
        - Provide clear reasoning context
        """
        
        # Extract object descriptions
        objects_desc = "\n".join([
            f"- {obj['id']}: {obj['description']}"
            for obj in vision_output['objects']
        ])
        
        # Format existing bindings
        existing_bindings = "\n".join([
            f"- {obj_id}: {data['semantic_label']} (confidence: {data['confidence']:.2f})"
            for obj_id, data in self.binding_table.items()
            if data['status'] == 'active'
        ]) or "- (none yet)"
        
        # Format recent history (last 3 actions)
        recent_history = "\n".join([
            f"- Step {h['step']}: {h['action']['semantic']['purpose']}"
            for h in history[-3:]
        ]) if history else "- (procedure just started)"
        
        # Build prompt
        prompt = f"""You are a laboratory semantic binding agent. Your task is to map physical objects to semantic concepts based on procedure steps and observed actions.

PROCEDURE STEP: {step}

ACTION OBSERVED:
Type: {action['type']}
Source object: {action['source']}
Destination object: {action['destination']}
Tool used: {action.get('tool', 'unknown')}

OBJECTS IN SCENE:
{objects_desc}

EXISTING BINDINGS:
{existing_bindings}

RECENT PROCEDURE HISTORY:
{recent_history}

TASK:
1. Identify which semantic entity from the procedure step corresponds to the source object
2. Identify which semantic entity corresponds to the destination object
3. Assign confidence scores (0.0-1.0) based on:
   - Direct mention in procedure step (high confidence)
   - Consistency with existing bindings (medium-high)
   - Visual features matching expectations (medium)
   - Spatial or temporal context (low-medium)

Return ONLY valid JSON in this format:
{{
  "bindings": [
    {{
      "object": "object_id",
      "label": "semantic label describing contents/purpose",
      "confidence": 0.XX,
      "reasoning": "brief explanation"
    }}
  ]
}}

JSON:"""
        
        return prompt
    
    def _infer_bindings(self, prompt: str) -> Dict:
        """
        Run LLM inference to generate semantic bindings.
        
        Returns:
            Dict with bindings and reasoning
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for consistency
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse JSON from response
        try:
            # Extract JSON (model may add text before/after)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                # No JSON found
                return {"bindings": []}
        
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response as JSON: {e}")
            print(f"Response was: {response[:200]}...")
            return {"bindings": []}
    
    def _update_binding_table(self, bindings_result: Dict):
        """
        Update persistent binding table with new bindings.
        """
        for binding in bindings_result.get('bindings', []):
            obj_id = binding['object']
            
            # Check if this is an update or new binding
            if obj_id in self.binding_table:
                # Update existing binding
                old_conf = self.binding_table[obj_id]['confidence']
                new_conf = binding['confidence']
                
                # Average confidence if updating
                avg_conf = (old_conf + new_conf) / 2
                
                self.binding_table[obj_id].update({
                    'semantic_label': binding['label'],
                    'confidence': avg_conf,
                    'last_updated': time.time(),
                    'update_count': self.binding_table[obj_id].get('update_count', 0) + 1
                })
            else:
                # New binding
                self.binding_table[obj_id] = {
                    'semantic_label': binding['label'],
                    'confidence': binding['confidence'],
                    'assigned_at': time.time(),
                    'first_seen': time.time(),
                    'last_updated': time.time(),
                    'status': 'active',
                    'update_count': 0
                }
    
    def _enhance_output(self, vision_output: Dict, bindings_result: Dict) -> Dict:
        """
        Add semantic labels to vision output.
        """
        enhanced = vision_output.copy()
        
        # Add semantic labels to each object
        for obj in enhanced['objects']:
            obj_id = obj['id']
            
            if obj_id in self.binding_table:
                binding = self.binding_table[obj_id]
                obj['semantic_label'] = binding['semantic_label']
                obj['semantic_confidence'] = binding['confidence']
            else:
                obj['semantic_label'] = None
                obj['semantic_confidence'] = 0.0
        
        # Create semantic scene description
        labeled_objects = []
        for obj in enhanced['objects']:
            if obj['semantic_label']:
                labeled_objects.append(
                    f"{obj['semantic_label']} ({obj['id']})"
                )
            else:
                labeled_objects.append(
                    f"{obj['description']} ({obj['id']})"
                )
        
        enhanced['semantic_scene_description'] = (
            f"Laboratory workspace with {', '.join(labeled_objects)}."
        )
        
        # Add binding metadata
        enhanced['semantic_metadata'] = {
            'bindings_applied': len([o for o in enhanced['objects'] if o['semantic_label']]),
            'average_confidence': self._calculate_avg_confidence(enhanced['objects']),
            'binding_table_size': len(self.binding_table)
        }
        
        return enhanced
    
    def _add_existing_bindings(self, vision_output: Dict) -> Dict:
        """
        Fast path: Just add existing semantic labels without LLM inference.
        Used when no action is detected.
        """
        enhanced = vision_output.copy()
        
        for obj in enhanced['objects']:
            obj_id = obj['id']
            if obj_id in self.binding_table:
                binding = self.binding_table[obj_id]
                obj['semantic_label'] = binding['semantic_label']
                obj['semantic_confidence'] = binding['confidence']
            else:
                obj['semantic_label'] = None
                obj['semantic_confidence'] = 0.0
        
        return enhanced
    
    def _calculate_avg_confidence(self, objects: List[Dict]) -> float:
        """Calculate average confidence of semantic bindings."""
        confidences = [
            obj['semantic_confidence']
            for obj in objects
            if obj['semantic_confidence'] > 0
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def get_binding(self, object_id: str) -> Optional[Dict]:
        """Get semantic binding for an object."""
        return self.binding_table.get(object_id)
    
    def list_bindings(self) -> Dict:
        """Return all current bindings."""
        return self.binding_table.copy()
    
    def clear_bindings(self):
        """Clear all bindings (e.g., start of new procedure)."""
        self.binding_table = {}
    
    def save_bindings(self, filepath: str):
        """Save binding table to file."""
        with open(filepath, 'w') as f:
            json.dump(self.binding_table, f, indent=2, default=str)
    
    def load_bindings(self, filepath: str):
        """Load binding table from file."""
        with open(filepath, 'r') as f:
            self.binding_table = json.load(f)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'total_inferences': len(self.inference_times),
            'avg_time_ms': sum(self.inference_times) / len(self.inference_times) * 1000,
            'min_time_ms': min(self.inference_times) * 1000,
            'max_time_ms': max(self.inference_times) * 1000
        }
```

### 2. ActionDetector Class

**Location**: `core/reasoning/action_detector.py`

**Implementation**:

```python
import numpy as np
from typing import Dict, List, Optional
import time

class ActionDetector:
    """
    Detects user actions by comparing consecutive frames.
    
    Focuses on laboratory-specific actions:
    - Liquid transfers (pipetting)
    - Object movements
    - Equipment interactions
    """
    
    def __init__(
        self,
        movement_threshold: float = 100.0,  # pixels
        confidence_threshold: float = 0.7
    ):
        """
        Initialize action detector.
        
        Args:
            movement_threshold: Minimum pixel distance to consider movement
            confidence_threshold: Minimum confidence for action detection
        """
        self.movement_threshold = movement_threshold
        self.confidence_threshold = confidence_threshold
        
        # Store previous frame for comparison
        self.previous_objects = None
        self.previous_timestamp = None
        
        # Action history for sequence detection
        self.action_history = []
        
    def detect(self, current_objects: List[Dict]) -> Optional[Dict]:
        """
        Detect action between previous and current frame.
        
        Args:
            current_objects: List of detected objects in current frame
            
        Returns:
            Action dict or None if no action detected
        """
        current_time = time.time()
        
        # First frame - just store
        if self.previous_objects is None:
            self.previous_objects = current_objects
            self.previous_timestamp = current_time
            return None
        
        # Detect action
        action = self._detect_transfer_action(
            self.previous_objects,
            current_objects
        )
        
        # Update state
        self.previous_objects = current_objects
        self.previous_timestamp = current_time
        
        if action:
            self.action_history.append(action)
            # Keep last 10 actions
            if len(self.action_history) > 10:
                self.action_history.pop(0)
        
        return action
    
    def _detect_transfer_action(
        self,
        prev_objects: List[Dict],
        curr_objects: List[Dict]
    ) -> Optional[Dict]:
        """
        Detect liquid transfer action (pipetting).
        
        Strategy:
        1. Find pipette in both frames
        2. Check if pipette moved significantly
        3. Determine source (where pipette was)
        4. Determine destination (where pipette is now)
        """
        # Find pipette in both frames
        pipette_prev = self._find_object_by_class(prev_objects, "pipette")
        pipette_curr = self._find_object_by_class(curr_objects, "pipette")
        
        if not pipette_prev or not pipette_curr:
            return None  # No pipette detected
        
        # Calculate pipette movement
        distance = self._calculate_distance(
            pipette_prev['bbox'],
            pipette_curr['bbox']
        )
        
        if distance < self.movement_threshold:
            return None  # Pipette didn't move enough
        
        # Find source object (where pipette was)
        source = self._find_nearest_object(
            pipette_prev,
            prev_objects,
            exclude_class='pipette'
        )
        
        # Find destination object (where pipette is now)
        destination = self._find_nearest_object(
            pipette_curr,
            curr_objects,
            exclude_class='pipette'
        )
        
        if not source or not destination:
            return None  # Can't determine source/destination
        
        if source['id'] == destination['id']:
            return None  # Same object, not a transfer
        
        # Action detected!
        return {
            'type': 'transfer',
            'source': source['id'],
            'destination': destination['id'],
            'tool': 'pipette',
            'confidence': self._calculate_action_confidence(
                pipette_prev, pipette_curr, source, destination
            ),
            'timestamp': time.time(),
            'details': {
                'distance_moved': distance,
                'source_description': source.get('description'),
                'destination_description': destination.get('description')
            }
        }
    
    def _find_object_by_class(
        self,
        objects: List[Dict],
        class_name: str
    ) -> Optional[Dict]:
        """Find first object matching class name."""
        for obj in objects:
            if obj['class'] == class_name:
                return obj
        return None
    
    def _calculate_distance(self, bbox1: List, bbox2: List) -> float:
        """Calculate distance between bbox centers (pixels)."""
        # bbox format: [x, y, w, h]
        c1_x = bbox1[0] + bbox1[2] / 2
        c1_y = bbox1[1] + bbox1[3] / 2
        c2_x = bbox2[0] + bbox2[2] / 2
        c2_y = bbox2[1] + bbox2[3] / 2
        
        return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    def _find_nearest_object(
        self,
        reference: Dict,
        objects: List[Dict],
        exclude_class: Optional[str] = None
    ) -> Optional[Dict]:
        """Find object nearest to reference object."""
        min_distance = float('inf')
        nearest = None
        
        for obj in objects:
            if exclude_class and obj['class'] == exclude_class:
                continue
            
            distance = self._calculate_distance(
                reference['bbox'],
                obj['bbox']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest = obj
        
        return nearest
    
    def _calculate_action_confidence(
        self,
        pipette_prev: Dict,
        pipette_curr: Dict,
        source: Dict,
        destination: Dict
    ) -> float:
        """
        Calculate confidence score for detected action.
        
        Factors:
        - Distance moved (larger = more confident)
        - Source/destination proximity (closer = more confident)
        - Object detection confidence
        """
        # Base confidence from object detections
        conf = min(
            pipette_prev.get('confidence', 1.0),
            pipette_curr.get('confidence', 1.0),
            source.get('confidence', 1.0),
            destination.get('confidence', 1.0)
        )
        
        # Boost if movement is significant
        distance = self._calculate_distance(
            pipette_prev['bbox'],
            pipette_curr['bbox']
        )
        if distance > 200:
            conf = min(1.0, conf + 0.1)
        
        # Boost if source/destination are clear (close proximity)
        source_dist = self._calculate_distance(
            pipette_prev['bbox'],
            source['bbox']
        )
        dest_dist = self._calculate_distance(
            pipette_curr['bbox'],
            destination['bbox']
        )
        
        if source_dist < 50 and dest_dist < 50:
            conf = min(1.0, conf + 0.1)
        
        return round(conf, 2)
    
    def get_recent_actions(self, n: int = 5) -> List[Dict]:
        """Get last N detected actions."""
        return self.action_history[-n:]
    
    def clear_history(self):
        """Clear action history."""
        self.action_history = []
```

### 3. Integration Example

**Location**: `lab_vision_system.py` (modified)

```python
from core.reasoning.semantic_binding_agent import SemanticBindingAgent
from core.reasoning.action_detector import ActionDetector

class LabVisionSystem:
    def __init__(self, config):
        # Existing components
        self.vision_pipeline = VisionPipeline(config)
        self.workspace_memory = WorkspaceMemory(config)
        self.procedure_context = ProcedureContext(config)
        
        # NEW: Semantic binding components
        if config.get('enable_semantic_binding', True):
            self.semantic_binding = SemanticBindingAgent(
                model_name=config.get('semantic_binding_model', 'meta-llama/Llama-3.2-3B-Instruct'),
                device=config.get('device', 'auto'),
                quantization=config.get('quantization', None)
            )
            self.action_detector = ActionDetector(
                movement_threshold=config.get('movement_threshold', 100.0),
                confidence_threshold=config.get('action_confidence_threshold', 0.7)
            )
        else:
            self.semantic_binding = None
            self.action_detector = None
        
        # Procedure state
        self.current_procedure_step = None
    
    def process_frame(self, frame, imu_data):
        """
        Main frame processing with semantic binding.
        """
        # STEP 1: Vision processing (existing)
        vision_output = self.vision_pipeline.process(frame)
        
        # STEP 2: Semantic binding (NEW)
        if self.semantic_binding:
            # Detect action
            action = self.action_detector.detect(vision_output['objects'])
            
            # Enhance with semantic labels
            enhanced_output = self.semantic_binding.process(
                vision_output,
                action,
                self.current_procedure_step or "Observing laboratory workspace",
                self.procedure_context.get_recent_history(n=5)
            )
        else:
            # Semantic binding disabled - use raw vision output
            enhanced_output = vision_output
        
        # STEP 3: Workspace memory (existing, receives enhanced output)
        workspace_result = self.workspace_memory.process(
            enhanced_output,
            imu_data
        )
        
        # STEP 4: Procedure logging (existing, receives enhanced output)
        if action:
            self.procedure_context.log_action(
                enhanced_output,
                action,
                workspace_result['current_workspace']
            )
        
        # STEP 5: AI guidance (existing)
        guidance = self.generate_guidance(
            enhanced_output,
            workspace_result,
            self.procedure_context.get_full_log()
        )
        
        return {
            'vision': enhanced_output,
            'workspace': workspace_result,
            'guidance': guidance,
            'action': action
        }
    
    def set_procedure_step(self, step_text: str):
        """Update current procedure step (called by main AI)."""
        self.current_procedure_step = step_text
```

---

## Data Structures

### Vision Output (Enhanced)

```json
{
    "timestamp": "2025-03-16T10:08:15Z",
    "frame_id": 1234,
    
    "objects": [
        {
            "id": "test_tube_1",
            "class": "test tube",
            "bbox": [120, 200, 180, 350],
            "mask": "...",
            "confidence": 0.92,
            "description": "test tube with yellow liquid",
            
            "semantic_label": "compound A stock solution",
            "semantic_confidence": 0.92,
            "contains": ["compound_A"],
            "status": "active"
        },
        {
            "id": "petri_dish_1",
            "class": "petri dish",
            "bbox": [450, 300, 550, 380],
            "mask": "...",
            "confidence": 0.89,
            "description": "petri dish with clear solution",
            
            "semantic_label": "solution B with compound A added",
            "semantic_confidence": 0.95,
            "contains": ["solution_B", "compound_A_200uL"],
            "status": "active"
        }
    ],
    
    "scene_description": "Laboratory bench with test tube on left, petri dish center",
    
    "semantic_scene_description": "Laboratory workspace with compound A stock solution (test_tube_1), solution B with compound A added (petri_dish_1).",
    
    "semantic_metadata": {
        "bindings_applied": 2,
        "average_confidence": 0.935,
        "binding_table_size": 5
    }
}
```

### Action Detection Result

```json
{
    "type": "transfer",
    "source": "test_tube_1",
    "destination": "petri_dish_1",
    "tool": "pipette",
    "confidence": 0.85,
    "timestamp": "2025-03-16T10:08:15.234Z",
    "details": {
        "distance_moved": 245.7,
        "source_description": "test tube with yellow liquid",
        "destination_description": "petri dish with clear solution"
    }
}
```

### Enhanced Procedure Log Entry

```json
{
    "step": 4,
    "timestamp": "2025-03-16T10:08:15Z",
    "workspace": "lab_bench_a",
    
    "action": {
        "type": "transfer",
        
        "physical": {
            "source": "test_tube_1",
            "source_description": "test tube with yellow liquid",
            "destination": "petri_dish_1",
            "destination_description": "petri dish with clear solution",
            "tool": "pipette_p200",
            "volume_visual": "~200µL"
        },
        
        "semantic": {
            "source_label": "compound A stock solution",
            "source_confidence": 0.92,
            "destination_label": "solution B",
            "destination_confidence": 0.95,
            "purpose": "add compound A to solution B per protocol step 4",
            "expected_result": "solution B should contain compound A"
        },
        
        "binding_confidence": 0.93
    },
    
    "observation": {
        "color_change": "none",
        "temperature": "23°C",
        "user_notes": "No visible reaction initially"
    },
    
    "result": {
        "physical": "petri_dish_1 now contains mixture of two liquids",
        "semantic": "solution B now contains compound A (200µL added)",
        "objects_updated": [
            {
                "object": "test_tube_1",
                "new_status": "depleted (some volume remaining)"
            },
            {
                "object": "petri_dish_1",
                "new_contents": ["solution_B_original", "compound_A_200uL"]
            }
        ]
    }
}
```

---

## Workflow Examples

### Example 1: Simple Transfer

**Scenario**: Add compound A to solution B

```
Frame N-1:
  Objects: test_tube_1 (left), petri_dish_1 (center), pipette (right)

Frame N:
  Objects: test_tube_1 (left), petri_dish_1 (center), pipette (center, near petri_dish)
  
Action Detector:
  → Pipette moved 250 pixels
  → Was near test_tube_1, now near petri_dish_1
  → ACTION: transfer from test_tube_1 to petri_dish_1

Semantic Binding Agent receives:
  Procedure Step: "Pipette 200µL of compound A into solution B"
  Action: transfer from test_tube_1 to petri_dish_1
  Existing Bindings: (none)

LLM Reasoning:
  "Procedure mentions 'compound A' as source.
   User transferred from test_tube_1.
   → test_tube_1 = compound A
   
   Procedure mentions 'solution B' as destination.
   User transferred to petri_dish_1.
   → petri_dish_1 = solution B"

Bindings Created:
  test_tube_1 → "compound A stock solution" (confidence: 0.92)
  petri_dish_1 → "solution B with compound A added" (confidence: 0.95)

Enhanced Output:
  "Transferred 200µL from compound A stock solution (test_tube_1)
   to solution B (petri_dish_1). Solution B now contains compound A."

Procedure Log:
  [10:08, lab_bench_a] Transferred 200µL compound A (test_tube_1) 
                       to solution B (petri_dish_1). No color change observed.
```

### Example 2: Ambiguous Objects

**Scenario**: Multiple test tubes, need to infer which is which

```
Frame: 
  Objects: test_tube_1 (left), test_tube_2 (right), petri_dish_1 (center)
  
Procedure Step: "Add compound A to solution B"

Action: transfer from test_tube_2 to petri_dish_1

Existing Bindings:
  petri_dish_1 → "solution B" (from previous step)

Semantic Binding Agent receives:
  Procedure Step: "Add compound A to solution B"
  Action: transfer from test_tube_2 to petri_dish_1
  Bindings: petri_dish_1 = solution B

LLM Reasoning:
  "Two test tubes visible: test_tube_1 and test_tube_2.
   User selected test_tube_2 for transfer.
   Procedure mentions compound A.
   → test_tube_2 likely = compound A
   
   Confidence: MEDIUM (0.75) because:
   - Temporal alignment: yes (procedure step matches action)
   - Object eliminated: test_tube_1 was not selected
   - No visual features to confirm (both tubes similar)
   
   Note: User's choice of test_tube_2 suggests confidence.
   If incorrect, user would have used test_tube_1."

Binding Created:
  test_tube_2 → "compound A stock solution" (confidence: 0.75)

Fallback: If confidence < 0.8, system flags for verification:
  "I think test_tube_2 contains compound A (75% confident).
   Can you confirm this is compound A?"
```

### Example 3: Compound Tracking

**Scenario**: Track compound through multiple transfers

```
Step 1:
  Action: Prepare compound A in test_tube_1
  Binding: test_tube_1 → "compound A stock solution"

Step 2:
  Action: Transfer from test_tube_1 to petri_dish_1 (solution B)
  Update: petri_dish_1 → "solution B with compound A"
          test_tube_1 → "compound A stock (partially depleted)"

Step 3:
  Action: Transfer from petri_dish_1 to centrifuge_tube_1
  Reasoning: "petri_dish_1 contains solution B + compound A.
              Transfer destination is centrifuge_tube_1.
              → centrifuge_tube_1 now has solution B + compound A"
  Binding: centrifuge_tube_1 → "solution B + compound A for centrifugation"

Step 4:
  Procedure: "After centrifugation, collect supernatant"
  Action: Transfer from centrifuge_tube_1 to clean_tube_1
  Reasoning: "Centrifugation separates components.
              Collecting supernatant means taking liquid portion.
              → clean_tube_1 has supernatant (solution B + compound A)"
  Binding: clean_tube_1 → "supernatant (solution B + compound A)"
          centrifuge_tube_1 → "pellet (cell debris)"

Main AI can now reason:
  "Compound A has been: prepared → added to solution B → 
   centrifuged → collected as supernatant. Compound A is now in 
   clean_tube_1 (the supernatant)."
```

---

## Performance Optimization

### 1. Conditional Execution

**Strategy**: Only run semantic binding when actions detected

```python
# In process_frame()
action = self.action_detector.detect(vision_output['objects'])

if action:
    # Action detected - run semantic binding (50-200ms)
    enhanced = self.semantic_binding.process(...)
else:
    # No action - fast path (< 5ms)
    enhanced = self.semantic_binding.add_existing_labels(vision_output)
```

**Impact**:
- Actions occur ~10-20% of frames
- Average overhead: (0.2 × 150ms) + (0.8 × 2ms) = 31.6ms
- vs. always running: 150ms
- **Speedup: 4.7x**

### 2. Prompt Optimization

**Short prompts = faster inference**

```python
# BAD: Verbose prompt (200 tokens → 150ms inference)
prompt = """
You are a highly intelligent laboratory semantic binding agent designed
to map physical objects detected by a computer vision system to semantic
concepts mentioned in laboratory procedures. Your task is to carefully
analyze the following information and make informed decisions...
[... 150 more words ...]
"""

# GOOD: Concise prompt (80 tokens → 80ms inference)
prompt = """
Map physical objects to procedure entities.

PROCEDURE: {step}
ACTION: transfer from {source} to {dest}
OBJECTS: {objects}
BINDINGS: {bindings}

Return JSON: {{"bindings": [{{"object": "...", "label": "...", "confidence": 0.XX}}]}}
"""

# Impact: 47% faster inference
```

### 3. Caching

**Cache identical scenarios**

```python
class SemanticBindingAgent:
    def __init__(self):
        self.cache = {}  # {cache_key: result}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def process(self, vision_output, action, procedure_step):
        # Generate cache key
        cache_key = self._generate_cache_key(action, procedure_step)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Cache miss - run inference
        self.cache_misses += 1
        result = self._infer_bindings(...)
        
        # Store in cache
        self.cache[cache_key] = result
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entry
            oldest = list(self.cache.keys())[0]
            del self.cache[oldest]
        
        return result
```

**Impact**:
- Similar procedures have ~30% cache hit rate
- Cached responses: <1ms vs 150ms
- **Effective speedup: 1.3x**

### 4. Batch Processing

**Process multiple actions together** (for offline analysis)

```python
def process_batch(self, actions_list):
    """
    Process multiple actions in one LLM call.
    More efficient for offline/batch processing.
    """
    # Build prompt with all actions
    prompt = f"""
    Analyze these {len(actions_list)} laboratory actions:
    
    {format_actions(actions_list)}
    
    Return JSON array of bindings for each action.
    """
    
    # Single inference for all actions
    result = self._infer_bindings(prompt)
    
    return result

# Impact: N actions in 200ms vs N × 150ms
# For N=5: 200ms vs 750ms → 3.75x faster
```

### 5. Model Quantization

**8-bit vs 16-bit precision**

```python
# FP16 (default):
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16
)
# Inference: 150ms, VRAM: 7GB

# 8-bit quantization:
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    load_in_8bit=True
)
# Inference: 170ms (+13% slower), VRAM: 3.5GB (-50%)

# Trade-off: Slight speed decrease for significant memory savings
```

---

## Error Handling

### 1. LLM Output Parsing Failures

```python
def _infer_bindings(self, prompt):
    try:
        response = self.model.generate(...)
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # LLM didn't return valid JSON
        print(f"Warning: Failed to parse LLM response: {response[:100]}")
        
        # Fallback: Extract partial information
        partial = self._extract_partial_bindings(response)
        if partial:
            return partial
        
        # Last resort: Return empty bindings
        return {"bindings": []}

def _extract_partial_bindings(self, response):
    """
    Try to extract semantic labels even if JSON is malformed.
    Use regex to find patterns like: test_tube_1 = compound A
    """
    import re
    pattern = r"(\w+)\s*(?:=|:|-?>)\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, response)
    
    if matches:
        return {
            "bindings": [
                {"object": obj, "label": label, "confidence": 0.7}
                for obj, label in matches
            ]
        }
    
    return None
```

### 2. Low Confidence Bindings

```python
def _update_binding_table(self, bindings_result):
    for binding in bindings_result.get('bindings', []):
        confidence = binding['confidence']
        
        if confidence < 0.5:
            # Very low confidence - don't create binding
            print(f"Warning: Low confidence binding rejected: {binding}")
            continue
        
        elif confidence < 0.7:
            # Medium confidence - flag for review
            binding['status'] = 'uncertain'
            binding['requires_verification'] = True
            print(f"Uncertain binding created: {binding['label']} (conf: {confidence})")
        
        else:
            # High confidence - proceed normally
            binding['status'] = 'active'
        
        self.binding_table[binding['object']] = binding
```

### 3. Contradictory Bindings

```python
def _detect_contradictions(self, new_binding, existing_binding):
    """
    Check if new binding contradicts existing binding.
    
    Example contradiction:
    - Existing: test_tube_1 = "compound A"
    - New: test_tube_1 = "solution B"
    """
    if new_binding['semantic_label'] != existing_binding['semantic_label']:
        # Labels differ - potential contradiction
        
        # Check if this could be valid (contents changed)
        if self._is_container_evolution(existing_binding, new_binding):
            # Valid: e.g., "solution B" → "solution B with compound A"
            return False
        
        # True contradiction
        print(f"⚠️ Contradiction detected:")
        print(f"  Existing: {existing_binding['semantic_label']} (conf: {existing_binding['confidence']})")
        print(f"  New: {new_binding['semantic_label']} (conf: {new_binding['confidence']})")
        
        # Resolution: Keep higher confidence binding
        if new_binding['confidence'] > existing_binding['confidence']:
            print(f"  → Replacing with new binding")
            return True  # Replace
        else:
            print(f"  → Keeping existing binding")
            return False  # Keep existing
    
    return False

def _is_container_evolution(self, old, new):
    """
    Check if binding represents valid container evolution.
    E.g., "solution B" → "solution B + compound A"
    """
    old_label = old['semantic_label'].lower()
    new_label = new['semantic_label'].lower()
    
    # New label contains old label = evolution
    return old_label in new_label
```

### 4. Missing Procedure Context

```python
def process(self, vision_output, action, procedure_step, history):
    # Guard against missing procedure step
    if not procedure_step or procedure_step.strip() == "":
        # No procedure context - can't do semantic binding
        print("Warning: No procedure step provided. Skipping semantic binding.")
        return self._add_existing_bindings(vision_output)
    
    # Guard against malformed action
    if action and not all(k in action for k in ['source', 'destination']):
        print(f"Warning: Malformed action: {action}")
        return self._add_existing_bindings(vision_output)
    
    # Proceed with semantic binding
    ...
```

---

## Testing Strategy

### Unit Tests

**Test File**: `tests/test_semantic_binding.py`

```python
import unittest
from core.reasoning.semantic_binding_agent import SemanticBindingAgent

class TestSemanticBindingAgent(unittest.TestCase):
    def setUp(self):
        # Use small model for testing
        self.agent = SemanticBindingAgent(
            model_name="meta-llama/Llama-3.2-1B-Instruct"
        )
    
    def test_simple_binding(self):
        """Test basic object-to-concept mapping."""
        vision_output = {
            'objects': [
                {'id': 'test_tube_1', 'description': 'test tube with yellow liquid'},
                {'id': 'petri_dish_1', 'description': 'petri dish with clear liquid'}
            ]
        }
        
        action = {
            'type': 'transfer',
            'source': 'test_tube_1',
            'destination': 'petri_dish_1'
        }
        
        procedure_step = "Pipette 200µL of compound A into solution B"
        
        result = self.agent.process(vision_output, action, procedure_step)
        
        # Check bindings created
        self.assertIn('test_tube_1', self.agent.binding_table)
        self.assertIn('compound A', self.agent.binding_table['test_tube_1']['semantic_label'].lower())
    
    def test_binding_persistence(self):
        """Test that bindings persist across frames."""
        # First frame - create binding
        ...
        
        # Second frame - no action, should still have binding
        vision_output_2 = {
            'objects': [{'id': 'test_tube_1', 'description': 'test tube'}]
        }
        result = self.agent.process(vision_output_2, None, "Observe")
        
        # Check binding still exists
        self.assertIn('test_tube_1', self.agent.binding_table)
    
    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        ...
        
        binding = self.agent.get_binding('test_tube_1')
        self.assertGreaterEqual(binding['confidence'], 0.0)
        self.assertLessEqual(binding['confidence'], 1.0)
```

### Integration Tests

**Test File**: `tests/test_semantic_integration.py`

```python
def test_end_to_end_workflow():
    """
    Test complete workflow: vision → action → semantic → log
    """
    system = LabVisionSystem(test_config)
    
    # Simulate procedure
    system.set_procedure_step("Pipette 200µL of compound A into solution B")
    
    # Frame 1: pipette near test_tube
    frame1 = load_test_frame('frame_001.jpg')
    result1 = system.process_frame(frame1, mock_imu_data)
    
    # Frame 2: pipette moved to petri_dish
    frame2 = load_test_frame('frame_002.jpg')
    result2 = system.process_frame(frame2, mock_imu_data)
    
    # Check action detected
    assert result2['action'] is not None
    assert result2['action']['type'] == 'transfer'
    
    # Check semantic labels added
    enhanced = result2['vision']
    test_tube = [o for o in enhanced['objects'] if o['id'] == 'test_tube_1'][0]
    assert test_tube['semantic_label'] is not None
    assert 'compound A' in test_tube['semantic_label'].lower()
    
    # Check procedure log
    log_entry = system.procedure_context.get_latest_entry()
    assert 'compound A' in log_entry['action']['semantic']['source_label']
```

### Performance Tests

```python
def test_latency():
    """Ensure semantic binding meets latency requirements."""
    agent = SemanticBindingAgent()
    
    # Run 100 inferences
    times = []
    for i in range(100):
        start = time.time()
        result = agent.process(test_vision_output, test_action, test_step)
        elapsed = time.time() - start
        times.append(elapsed)
    
    # Check metrics
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    assert avg_time < 0.200, f"Average time too high: {avg_time:.3f}s"
    assert max_time < 0.500, f"Max time too high: {max_time:.3f}s"
    
    print(f"✓ Latency test passed: avg={avg_time*1000:.1f}ms, max={max_time*1000:.1f}ms")
```

---

## Configuration

### Config File: `config/semantic_binding_config.yaml`

```yaml
semantic_binding:
  # Enable/disable semantic binding system
  enabled: true
  
  # Model settings
  model:
    name: "meta-llama/Llama-3.2-3B-Instruct"
    device: "auto"  # "cuda", "cpu", or "auto"
    quantization: null  # null, "8bit", or "4bit"
    torch_dtype: "float16"
  
  # Performance settings
  performance:
    max_new_tokens: 512
    temperature: 0.1
    do_sample: false
    batch_size: 1
  
  # Action detection
  action_detection:
    enabled: true
    movement_threshold: 100.0  # pixels
    confidence_threshold: 0.7
  
  # Binding management
  binding:
    min_confidence_threshold: 0.5  # Reject below this
    uncertain_threshold: 0.7  # Flag for review below this
    cache_size: 100
    enable_caching: true
  
  # Error handling
  error_handling:
    retry_on_parse_failure: true
    max_retries: 2
    fallback_to_existing_bindings: true
  
  # Logging
  logging:
    save_bindings: true
    bindings_file: "data/bindings/binding_table.json"
    log_level: "INFO"
    save_llm_prompts: false  # For debugging
    save_llm_responses: false  # For debugging
```

---

## Deployment Guide

### Step 1: Setup on RunPod

```bash
# SSH into RunPod instance
ssh root@<runpod-instance-ip>

# Navigate to project
cd lab-vision-system

# Pull latest code
git pull origin main

# Install semantic binding dependencies
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install sentencepiece==0.1.99

# Download Llama 3.2 3B model (one-time, ~6GB)
python scripts/download_semantic_binding_model.py
```

### Step 2: Verify Installation

```bash
python verify_semantic_binding.py
```

Expected output:
```
Verifying semantic binding system...
✓ Transformers installed: 4.36.0
✓ Model downloaded: Llama-3.2-3B-Instruct
✓ GPU available: NVIDIA RTX 4090
✓ VRAM available: 10.3 GB free
✓ Semantic binding agent initialized
✓ Test inference successful (142ms)
✓ Action detector initialized
✓ Integration test passed

Semantic binding system ready!
```

### Step 3: Update Configuration

Edit `config/memory_config.yaml`:

```yaml
semantic_binding:
  enabled: true  # Enable semantic binding
  model:
    name: "meta-llama/Llama-3.2-3B-Instruct"
    device: "auto"
    quantization: null  # Use "8bit" if VRAM constrained
```

### Step 4: Test with Sample Video

```bash
python process_video.py \
  --video videos/lab_procedure_demo.mp4 \
  --enable-semantic-binding \
  --save-bindings \
  --output output/semantic_binding_test/
```

Check output:
```
output/semantic_binding_test/
├── annotated_video.mp4           # Video with annotations
├── results.json                  # Detection results
├── binding_table.json            # Object-semantic mappings
├── semantic_procedure_log.json   # Enhanced procedure log
└── performance_stats.json        # Latency metrics
```

### Step 5: Production Deployment

```python
# In your main application
from lab_vision_system import LabVisionSystem

# Initialize with semantic binding enabled
config = {
    'enable_semantic_binding': True,
    'semantic_binding_model': 'meta-llama/Llama-3.2-3B-Instruct',
    'device': 'cuda',
    'quantization': None
}

system = LabVisionSystem(config)

# Process frames
while True:
    frame = glasses.get_frame()
    imu = glasses.get_imu()
    
    result = system.process_frame(frame, imu)
    
    # result['vision'] now has semantic labels
    # result['action'] contains detected actions
    # Binding table automatically maintained
```

---

## Summary

### What This System Provides

**Core Capability**: Maps physical objects to semantic concepts

**Key Features**:
- ✅ Lightweight (3B model, 50-200ms inference)
- ✅ Conditional (only runs on actions, ~10-20% of frames)
- ✅ Probabilistic (confidence scores for uncertainty)
- ✅ Persistent (bindings maintained across frames/workspaces)
- ✅ Self-correcting (updates based on observations)

**Integration**: Single interception point between vision and procedure logging

**Performance**: Average 30ms overhead (150ms × 20% action frames + 2ms × 80% no-action frames)

**VRAM**: 7GB for semantic binding + 5GB vision = 12GB total (fits RTX 4090)

### Benefits Over Physical-Only Tracking

| Aspect | Without Semantic Binding | With Semantic Binding |
|--------|-------------------------|----------------------|
| **Procedure Log** | "Liquid from tube_1 to dish_1" | "200µL compound A to solution B" |
| **AI Understanding** | Physical movements only | Chemical relationships |
| **Error Detection** | Cannot detect chemical errors | "Compound A added twice!" |
| **Cross-reference** | "What's in tube_1?" → unknown | "Compound A stock solution" |
| **Guidance Quality** | Generic instructions | Chemically-informed guidance |

### Next Steps

1. **Test on sample procedures** - Validate binding accuracy
2. **Tune confidence thresholds** - Optimize for your lab environment
3. **Expand equipment classes** - Add lab-specific semantic categories
4. **Monitor performance** - Track inference times and VRAM usage
5. **Iterate on prompts** - Improve LLM reasoning quality

---

**This semantic binding system transforms your lab vision system from a passive observer into an intelligent assistant that truly understands the chemistry happening in the lab.**