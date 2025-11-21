# TEP Implementation Checklist

Use this checklist to track your implementation progress. Check off items as you complete them.

---

## Phase 0: Setup & Preparation

- [ ] Review IMPLEMENTATION_GUIDE.md completely
- [ ] Review IMPLEMENTATION_SUMMARY.md for quick reference
- [ ] Review temporal_event_parser_enhanced.py for reference implementations
- [ ] Understand existing codebase (`process_video.py`, `adaptive_lab_system.py`)
- [ ] Verify `videos/DNA_Extraction.txt` exists
- [ ] Verify test video exists in `videos/`

**Time**: 30 minutes

---

## Phase 1: Directory Structure (15 minutes)

- [ ] Copy `setup_tep.py` to project root
- [ ] Run: `python setup_tep.py`
- [ ] Verify all directories created:
  - [ ] `src/`
  - [ ] `src/procedure/`
  - [ ] `src/tep/`
  - [ ] `src/integration/`
  - [ ] `templates/`
  - [ ] `tests/`
  - [ ] `scripts/`
- [ ] Verify all `__init__.py` files created
- [ ] Run: `python scripts/verify_setup.py`
- [ ] Update `requirements.txt` with new dependencies
- [ ] Run: `pip install -r requirements.txt`

**Checkpoint**: All directories exist, dependencies installed

---

## Phase 2: Data Structures (30 minutes)

### File: `src/tep/data_structures.py`

- [ ] Copy complete file from IMPLEMENTATION_GUIDE.md Phase 1, Step 1.2
- [ ] Implement `ActionType` enum (11 types)
- [ ] Implement `MismatchType` enum (5 types)
- [ ] Implement `BoundaryType` enum (3 types)
- [ ] Implement `FrameData` dataclass
- [ ] Implement `TemporalWindow` dataclass
- [ ] Implement `TEPEvent` dataclass
- [ ] Implement `ProcedureTemplate` dataclass

**Test**:
```bash
python -c "from src.tep.data_structures import ActionType, FrameData; print('✓ Imports work')"
```

- [ ] Test passes
- [ ] Can create FrameData instance
- [ ] Can create TEPEvent instance

**Checkpoint**: All data structures defined and importable

---

## Phase 3: Procedure Template Generator (2 hours)

### 3.1: Text Preprocessor (20 minutes)

**File**: `src/procedure/text_preprocessor.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 2, Step 2.1
- [ ] Implement `TextPreprocessor` class
- [ ] Implement `preprocess()` method
- [ ] Implement `_normalize_text()` method
- [ ] Implement `_tokenize()` method
- [ ] Implement `_identify_measurements()` method

**Test**:
```python
from src.procedure.text_preprocessor import TextPreprocessor
tp = TextPreprocessor()
result = tp.preprocess("Add 200µL of HCl to tube")
assert 'original' in result
assert 'normalized' in result
assert 'tokens' in result
print("✓ TextPreprocessor works")
```

- [ ] Test passes

### 3.2: Parameter Extractor (40 minutes)

**File**: `src/procedure/parameter_extractor.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 2, Step 2.2
- [ ] Implement `ParameterExtractor` class
- [ ] Implement `extract_all()` method
- [ ] Implement volume extraction (3 patterns)
- [ ] Implement temperature extraction (3 patterns)
- [ ] Implement duration extraction (4 patterns)
- [ ] Implement speed extraction (2 patterns)
- [ ] Implement count extraction (2 patterns)
- [ ] Implement concentration extraction (2 patterns)

**Test**:
```python
from src.procedure.parameter_extractor import ParameterExtractor
pe = ParameterExtractor()
result = pe.extract_all("Add 200µL at 37°C for 30 minutes at 10,000 rpm")
assert result['volume']['value'] == 200
assert result['temperature']['value'] == 37
assert result['duration']['value'] == 30
assert result['speed']['value'] == 10000
print("✓ ParameterExtractor works")
```

- [ ] Volume extraction test passes
- [ ] Temperature extraction test passes
- [ ] Duration extraction test passes
- [ ] Speed extraction test passes
- [ ] Count extraction test passes

### 3.3: Action Classifier (30 minutes)

**File**: `src/procedure/action_classifier.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 2, Step 2.3
- [ ] Implement `ActionClassifier` class
- [ ] Implement `classify()` method
- [ ] Implement `_disambiguate()` method
- [ ] Implement `_adjust_with_parameters()` method
- [ ] Implement `_infer_from_context()` method
- [ ] Implement `_explain_classification()` method

**Test**:
```python
from src.procedure.action_classifier import ActionClassifier
ac = ActionClassifier()

# Test transfer
result = ac.classify("Add 200µL to tube", {'volume': {'value': 200, 'unit': 'µL'}})
assert result['action_type'] == 'transfer'
assert result['confidence'] > 0.5

# Test mix
result = ac.classify("Mix by pipetting 10 times", {'count': {'value': 10}})
assert result['action_type'] == 'mix'

print("✓ ActionClassifier works")
```

- [ ] Transfer classification test passes
- [ ] Mix classification test passes
- [ ] Heat classification test passes
- [ ] Centrifuge classification test passes

### 3.4: Template Generator (30 minutes)

**File**: `src/procedure/template_generator.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 2, Step 2.4
- [ ] Implement `ProcedureTemplateGenerator` class
- [ ] Implement `generate_template()` method
- [ ] Implement `_process_step()` method
- [ ] Implement `_assemble_template()` method
- [ ] Implement `_estimate_duration()` method

**Test**:
```python
from src.procedure.template_generator import ProcedureTemplateGenerator
gen = ProcedureTemplateGenerator()

steps = [
    "Add 200µL compound A to solution B",
    "Mix by pipetting 10 times",
    "Incubate at 37°C for 30 minutes"
]

template = gen.generate_template(
    title="Test Procedure",
    user_id="test_user",
    step_descriptions=steps
)

assert len(template['steps']) == 3
assert template['steps'][0]['expected_action'] == 'transfer'
assert template['steps'][1]['expected_action'] == 'mix'
print("✓ Template Generator works")
```

- [ ] Template generation test passes
- [ ] All steps have action types
- [ ] Parameters extracted in template
- [ ] Duration estimation works

**Checkpoint**: Can generate complete procedure templates from text

---

## Phase 4: TEP Core Modules (3 hours)

### 4.1: Window Manager (45 minutes)

**File**: `src/tep/window_manager.py`

- [ ] Copy `TemporalWindowManager` from `temporal_event_parser_enhanced.py`
- [ ] Update imports to use `data_structures.py`
- [ ] Implement `update()` method
- [ ] Implement `_calculate_motion_metric()` method
- [ ] Implement `_detect_active_objects()` method
- [ ] Implement `_detect_action_boundary()` method
- [ ] Implement boundary detection methods:
  - [ ] `_detect_approach_withdraw_pattern()`
  - [ ] `_detect_oscillation_stop()`
  - [ ] `_detect_placement_stability()`
- [ ] Implement `_extract_window_with_uncertainty()` method

**Test**: Will test in integration (needs full pipeline)

- [ ] File compiles without syntax errors
- [ ] Imports work

### 4.2: Interaction Graph Builder (30 minutes)

**File**: `src/tep/graph_builder.py`

- [ ] Copy `InteractionGraphBuilder` from enhanced implementation
- [ ] Update imports
- [ ] Implement `build_graph()` method
- [ ] Implement `detect_multi_object_patterns()` method
- [ ] Implement `_detect_serial_transfer()` method
- [ ] Implement `_detect_parallel_pipetting()` method
- [ ] Implement `_compute_distance()` method
- [ ] Implement `_compute_alignment()` method

- [ ] File compiles without errors
- [ ] Imports work

### 4.3: Action Classifier (45 minutes)

**File**: `src/tep/action_classifier.py`

- [ ] Copy `RuleBasedActionClassifier` from enhanced implementation
- [ ] Update imports
- [ ] Implement `classify()` method
- [ ] Implement `_calculate_confidence()` method
- [ ] Implement `_extract_features()` method
- [ ] Implement feature extraction helpers:
  - [ ] `_has_tool()`
  - [ ] `_get_tool_trajectory()`
  - [ ] `_count_container_approaches()`
  - [ ] `_count_oscillations()`
  - [ ] `_detect_placement_events()`
  - [ ] `_analyze_motion_pattern()`
- [ ] Implement rule detection methods:
  - [ ] `_detect_transfer_pattern()`
  - [ ] `_detect_mix_pattern()`
  - [ ] `_detect_heat_pattern()`
  - [ ] `_detect_cool_pattern()`
  - [ ] `_detect_centrifuge_pattern()`
  - [ ] `_detect_wait_pattern()`
  - [ ] `_detect_vortex_pattern()`

- [ ] File compiles without errors
- [ ] Imports work

### 4.4: Deviation Handler (30 minutes)

**File**: `src/tep/deviation_handler.py`

- [ ] Copy `DeviationHandler` from enhanced implementation
- [ ] Update imports
- [ ] Implement `handle_mismatch()` method
- [ ] Implement `_find_matching_future_step()` method
- [ ] Implement `_is_reasonable_action()` method
- [ ] Implement `_check_lost_sync()` method
- [ ] Implement `record_user_correction()` method

- [ ] File compiles without errors
- [ ] Imports work

### 4.5: No-Action Detector (20 minutes)

**File**: `src/tep/no_action_detector.py`

- [ ] Copy `NoActionDetector` from enhanced implementation
- [ ] Update imports
- [ ] Implement `check_inactivity()` method
- [ ] Implement `_detect_workspace_exit()` method
- [ ] Implement `reset_timer()` method

- [ ] File compiles without errors
- [ ] Imports work

### 4.6: Main TEP Orchestrator (30 minutes)

**File**: `src/tep/temporal_event_parser.py`

- [ ] Copy `TemporalEventParser` from enhanced implementation
- [ ] Update all imports
- [ ] Implement `__init__()` method
- [ ] Implement `process_frame()` method
- [ ] Implement `_create_event()` method
- [ ] Implement `_create_inactivity_event()` method
- [ ] Implement `get_training_data()` method

**Test**:
```python
from src.tep.temporal_event_parser import TemporalEventParser
from src.integration.procedure_executor import ProcedureContext
print("✓ TEP imports successfully")
```

- [ ] Test passes
- [ ] All imports work

**Checkpoint**: All TEP modules implemented and importable

---

## Phase 5: Integration Layer (2 hours)

### 5.1: Frame Converter (1 hour)

**File**: `src/integration/frame_converter.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 4, Step 4.1
- [ ] Implement `FrameConverter` class
- [ ] Implement `convert_frame()` method
- [ ] Implement `_create_object()` method
- [ ] Implement `_get_or_assign_id()` method (object tracking)
- [ ] Implement `_calculate_iou()` method
- [ ] Implement `reset_tracking()` method

**CRITICAL STEP**: Adapt to your YOLO format

- [ ] Add debug code to `process_video.py`:
```python
# In process_video.py, add temporarily:
for result in self.process_video(video_path):
    print("YOLO output format:", type(result))
    print("Keys:", result.keys() if hasattr(result, 'keys') else 'N/A')
    if hasattr(result, '__dict__'):
        print("Attributes:", result.__dict__)
    break
```

- [ ] Run existing video processor to see format
- [ ] Document your actual format:
  ```
  My YOLO format:
  - Type: ___________
  - Detection structure: ___________
  - Bbox format: ___________
  ```
- [ ] Adjust `_create_object()` to match YOUR format
- [ ] Adjust `convert_frame()` to extract YOUR data correctly

**Test**:
```python
from src.integration.frame_converter import FrameConverter

converter = FrameConverter()

# Use YOUR actual YOLO format here
yolo_det = [/* your format */]

frame_data = converter.convert_frame(
    frame_number=1,
    timestamp=0.033,
    yolo_detections=yolo_det,
    sam_masks=[],
    clip_classifications=[]
)

assert len(frame_data.objects) > 0
print(f"✓ Created FrameData with {len(frame_data.objects)} objects")
```

- [ ] Test passes with YOUR format
- [ ] Objects have valid IDs
- [ ] Objects have valid positions
- [ ] Tracking works across frames

### 5.2: Procedure Executor (1 hour)

**File**: `src/integration/procedure_executor.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 4, Step 4.2
- [ ] Implement `ProcedureContext` class
- [ ] Implement `get_current_step()` method
- [ ] Implement `advance_step()` method
- [ ] Implement `jump_to_step()` method
- [ ] Implement `ProcedureExecutor` class
- [ ] Implement `process_frame()` method
- [ ] Implement `get_all_events()` method
- [ ] Implement `export_log()` method

**Test**:
```python
from src.integration.procedure_executor import ProcedureExecutor

# Use template from previous test
executor = ProcedureExecutor(template)

assert executor.procedure_context.current_step_number == 1
print(f"✓ Executor initialized on step 1/{executor.procedure_context.total_steps}")
```

- [ ] Test passes
- [ ] Can get current step
- [ ] Can advance steps
- [ ] Can export log

**Checkpoint**: Integration layer complete, can connect vision to TEP

---

## Phase 6: Test Script (1 hour)

**File**: `scripts/test_video_with_tep.py`

- [ ] Copy from IMPLEMENTATION_GUIDE.md Phase 5, Step 5.1
- [ ] Update imports to match YOUR system
- [ ] Adjust video processor integration:
  - [ ] Import YOUR `VideoProcessor` class
  - [ ] Match YOUR iterator/generator pattern
  - [ ] Extract YOUR detection format correctly
- [ ] Implement main() function:
  - [ ] Load procedure text file
  - [ ] Generate template
  - [ ] Initialize components
  - [ ] Process video frame-by-frame
  - [ ] Export results

**Test**: Syntax check
```bash
python -m py_compile scripts/test_video_with_tep.py
```

- [ ] No syntax errors
- [ ] All imports found

**Checkpoint**: Test script ready to run

---

## Phase 7: End-to-End Testing (1-2 hours)

### 7.1: Generate Template

```bash
python -c "
import json
from pathlib import Path
from src.procedure.template_generator import ProcedureTemplateGenerator

with open('videos/DNA_Extraction.txt') as f:
    steps = [line.strip() for line in f if line.strip()]

gen = ProcedureTemplateGenerator()
template = gen.generate_template(
    title='DNA Extraction Protocol',
    user_id='test_user',
    step_descriptions=steps
)

Path('templates').mkdir(exist_ok=True)
with open('templates/dna_extraction.json', 'w') as f:
    json.dump(template, f, indent=2)

print(f'✓ Generated template with {len(template[\"steps\"])} steps')
"
```

- [ ] Command runs without errors
- [ ] `templates/dna_extraction.json` created
- [ ] Template has correct number of steps

### 7.2: Inspect Template

```bash
cat templates/dna_extraction.json | python -m json.tool | head -50
```

- [ ] JSON is valid
- [ ] Steps have `expected_action`
- [ ] Steps have `parameters`
- [ ] Actions make sense for procedure

### 7.3: Run Video Processing

```bash
python scripts/test_video_with_tep.py 2>&1 | tee test_output.log
```

Watch for:
- [ ] Template loads successfully
- [ ] Video opens successfully
- [ ] Frames process (should see frame numbers)
- [ ] Events detected (should see event output)
- [ ] No crashes or exceptions
- [ ] Log file created

### 7.4: Check Results

```bash
# View execution log
cat output/execution_log.json | python -m json.tool

# Check statistics
python -c "
import json
log = json.load(open('output/execution_log.json'))
print(f'Steps: {log[\"completed_steps\"]}/{log[\"total_steps\"]}')
print(f'Events: {len(log[\"events\"])}')
for i, event in enumerate(log['events'][:5], 1):
    print(f'{i}. {event[\"action_type\"]}: {event[\"step_description\"][:50]}...')
"
```

Expected results:
- [ ] Log file exists
- [ ] Has events (at least a few)
- [ ] Events have action types
- [ ] Events linked to steps
- [ ] Some events match expectations

**Checkpoint**: End-to-end system works!

---

## Phase 8: Tuning & Validation (1-2 hours)

### 8.1: Accuracy Analysis

- [ ] Count matched vs mismatched events
- [ ] Calculate match percentage: ____%
- [ ] Identify common mismatch patterns

### 8.2: Parameter Tuning

If accuracy < 70%:

- [ ] Adjust window size in `window_manager.py`:
  ```python
  self.window_size = 3.0  # Try 2.0, 2.5, 3.0, 3.5
  ```

- [ ] Adjust confidence threshold in `action_classifier.py`:
  ```python
  if classification['confidence'] > 0.50:  # Try 0.50, 0.60, 0.70
  ```

- [ ] Check object tracking in `frame_converter.py`:
  ```python
  best_iou = 0.2  # Try 0.2, 0.25, 0.3
  ```

- [ ] Re-run test after each change
- [ ] Document what works best

### 8.3: Error Analysis

Common issues and fixes:

**No events detected**:
- [ ] Check: Are objects being detected? (print frame_data.objects)
- [ ] Check: Is window manager creating windows? (add debug prints)
- [ ] Fix: Lower thresholds or increase window size

**Wrong actions detected**:
- [ ] Check: Which actions are confused? (mix vs vortex, transfer vs discard)
- [ ] Fix: Adjust classification rules in `action_classifier.py`

**Poor tracking**:
- [ ] Check: Object IDs changing every frame?
- [ ] Fix: Improve IoU calculation or tracking logic

### 8.4: Final Validation

- [ ] Run full test 3 times
- [ ] Accuracy consistent across runs: ____%
- [ ] No crashes or errors
- [ ] Log files complete
- [ ] Events make sense for procedure

**Target**: 70-85% accuracy on first implementation

---

## Phase 9: Documentation & Cleanup

### 9.1: Code Documentation

- [ ] Add docstrings to all classes
- [ ] Add docstrings to all public methods
- [ ] Add inline comments for complex logic
- [ ] Update README.md with new features

### 9.2: Create Usage Examples

**File**: `examples/basic_usage.py`

- [ ] Show template generation
- [ ] Show video processing
- [ ] Show results analysis

### 9.3: Clean Up

- [ ] Remove debug print statements
- [ ] Remove commented-out code
- [ ] Fix any TODOs
- [ ] Format code consistently

### 9.4: Final Testing

- [ ] Run on different video
- [ ] Try different procedure text
- [ ] Verify nothing breaks

---

## Final Checklist

### Code Quality

- [ ] All files have proper imports
- [ ] All classes have docstrings
- [ ] No syntax errors
- [ ] No unused imports
- [ ] Code follows Python conventions

### Functionality

- [ ] Template generation works
- [ ] Video processing works
- [ ] TEP detects events
- [ ] Events validate against template
- [ ] Logs export correctly

### Testing

- [ ] Unit tests pass (if created)
- [ ] Integration test passes
- [ ] End-to-end test passes
- [ ] Accuracy ≥ 70%

### Documentation

- [ ] README updated
- [ ] Usage examples created
- [ ] Common issues documented
- [ ] Next steps documented

---

## Success Metrics

You've successfully implemented the system if:

✅ **Template Generation**:
- Can generate template from any text file
- 80%+ of parameters extracted correctly
- All steps have action classifications

✅ **Video Processing**:
- Processes video without crashes
- Detects 5+ events per minute of video
- Events have reasonable confidence scores

✅ **Accuracy**:
- 70%+ of events match expected actions
- 80%+ after parameter tuning
- Mismatches are flagged appropriately

✅ **Usability**:
- Single command to process video
- Clear output and logs
- Easy to understand results

---

## Time Tracking

Record your actual time spent:

- Setup: _____ hours
- Data Structures: _____ hours
- Template Generator: _____ hours
- TEP Core: _____ hours
- Integration: _____ hours
- Testing: _____ hours
- Tuning: _____ hours
- **Total**: _____ hours

**Target**: 9-10 hours

---

## Notes & Issues

Use this space to track issues and solutions:

```
Issue 1: [Description]
Solution: [What worked]

Issue 2: [Description]
Solution: [What worked]
```

---

## Completion Date

Started: ____________
Completed: ____________

**Congratulations!** 🎉

You've built a working Temporal Event Parser and Procedure Template Generator system!
