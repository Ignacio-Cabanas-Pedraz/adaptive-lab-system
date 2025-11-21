# Laboratory Terms Guide

This document lists laboratory-specific terms that can be added to the action classifier with instructional guides for students.

---

## Current Terms in System

These terms are already recognized by the action classifier:

| Term | Current Action | Keywords File Location |
|------|----------------|------------------------|
| resuspend | dissolve | `src/procedure/action_classifier.py` |
| reconstitute | dissolve | |
| aliquot | transfer | |
| vortex | mix | |
| autoclave | heat | |
| pellet | centrifuge | |
| equilibrate | wait | |

---

## Suggested Terms to Add

### Pipetting Techniques

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **aspirate** | Draw liquid into pipette tip by releasing plunger slowly | transfer | HIGH |
| **reverse pipette** | Technique for viscous/foaming liquids - overdraw then dispense to first stop | transfer | MEDIUM |
| **serial dilution** | Sequential dilutions (e.g., 1:10, 1:100, 1:1000) | transfer | HIGH |
| **trituration** | Repeated pipetting up and down to mix or break up cell clumps | mix | HIGH |

**Guide Example - Aspirate:**
```
1. Press plunger to first stop before inserting tip
2. Insert tip 2-3mm below liquid surface
3. Slowly release plunger to draw liquid
4. Wait 1 second before removing from liquid
5. Check tip for air bubbles
```

---

### Mixing/Homogenization

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **homogenize** | Make mixture uniform in consistency | mix | MEDIUM |
| **sonicate** | Use ultrasonic waves to mix, lyse cells, or shear DNA | mix | HIGH |
| **flick** | Flick tube with finger to mix small volumes | mix | HIGH |
| **tap** | Tap tube on bench to collect liquid at bottom | mix | HIGH |
| **pulse vortex** | Brief 1-2 second vortex bursts | mix | MEDIUM |

**Guide Example - Flick:**
```
1. Hold tube securely near the cap
2. Flick bottom of tube with finger
3. Repeat 3-5 times
4. Quick spin to collect liquid at bottom
```

---

### Separation Techniques

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **decant** | Carefully pour off liquid while retaining pellet | transfer | HIGH |
| **aspirate supernatant** | Use pipette to remove liquid above pellet | transfer | HIGH |
| **collect flowthrough** | Gather liquid that passes through column/filter | transfer | MEDIUM |
| **elute** | Wash bound substance off column/beads with buffer | transfer | HIGH |
| **fractionate** | Separate mixture into distinct portions | transfer | LOW |

**Guide Example - Decant:**
```
1. Note pellet position in tube
2. Slowly tilt tube away from pellet
3. Pour liquid into waste/collection container
4. Keep tube tilted until liquid is removed
5. Blot tube rim on clean paper towel
WARNING: Do not disturb the pellet
```

---

### Centrifugation

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **spin down** | Brief centrifugation to collect liquid/pellet | centrifuge | HIGH |
| **quick spin** | Very brief (5-10 sec) pulse spin | centrifuge | HIGH |
| **ultracentrifuge** | High-speed centrifugation (>100,000 x g) | centrifuge | LOW |

**Guide Example - Quick Spin:**
```
1. Balance tubes in centrifuge rotor
2. Close lid securely
3. Press pulse/start button
4. Allow to reach ~2000 rpm then stop
5. Wait for rotor to stop completely before opening
```

---

### Temperature Control

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **denature** | Heat to unfold proteins/separate DNA strands | heat | HIGH |
| **anneal** | Controlled cooling to allow binding/refolding | cool | HIGH |
| **quench** | Rapid cooling to stop reaction | cool | MEDIUM |
| **snap freeze** | Flash freeze in liquid nitrogen or dry ice | cool | MEDIUM |
| **thaw** | Bring frozen sample to working temperature | heat | HIGH |

**Guide Example - Snap Freeze:**
```
1. Prepare liquid nitrogen in dewar or dry ice in ethanol
2. Label cryovials clearly
3. Aliquot sample into cryovials
4. Immediately plunge into liquid nitrogen
5. Transfer to -80°C storage
SAFETY: Wear cryogenic gloves and face shield
```

---

### Sterilization/Safety

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **flame** | Sterilize instrument/tube opening with Bunsen burner | heat | HIGH |
| **UV sterilize** | Expose to UV light for decontamination | wait | LOW |
| **aseptic technique** | Sterile handling procedures | (instruction only) | HIGH |

**Guide Example - Flame:**
```
1. Turn on Bunsen burner to blue flame
2. Pass item through flame 2-3 times
3. Do not hold in flame (will overheat)
4. Allow to cool before use
5. Work near flame to maintain sterile field
SAFETY: Keep flammables away, tie back hair
```

---

### Molecular Biology Specific

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **electroporate** | Use electric pulse for cell transformation | transfer (or new) | MEDIUM |
| **transfect** | Introduce DNA/RNA into eukaryotic cells | transfer | MEDIUM |
| **lyse** | Break open cells to release contents | mix | HIGH |
| **extract** | Isolate specific component from mixture | transfer | HIGH |
| **precipitate** | Cause dissolved substance to form solid | wait | HIGH |
| **digest** | Enzymatic cutting (restriction enzymes, proteases) | wait | HIGH |
| **ligate** | Enzymatic joining of DNA fragments | wait | MEDIUM |

**Guide Example - Lyse (chemical):**
```
1. Add lysis buffer to sample
2. Mix by vortexing or pipetting
3. Incubate as specified (often on ice or 37°C)
4. Solution should become clear/viscous
5. Proceed immediately to next step
```

---

### Observation/Analysis

| Term | Description | Suggested Action | Priority |
|------|-------------|------------------|----------|
| **visualize** | Observe results (gel, microscope, plate) | measure | MEDIUM |
| **quantify** | Determine concentration/amount | measure | HIGH |
| **photograph** | Document results with camera | measure | LOW |
| **score** | Count colonies, bands, or evaluate results | measure | MEDIUM |

---

## Implementation Options

### Option 1: Add to Action Classifier
Add terms to `ACTION_KEYWORDS` in `src/procedure/action_classifier.py`

### Option 2: Technique Guide Database
Create separate database with detailed guides:
```python
TECHNIQUE_GUIDES = {
    'aspirate': {
        'action': 'transfer',
        'description': 'Draw liquid into pipette',
        'steps': [...],
        'tips': [...],
        'safety': [...],
        'common_errors': [...]
    }
}
```

### Option 3: Conversational Assistant Integration
Let Llama provide guides on-demand when student asks "How do I aspirate?"

---

## Decision Points

Before implementing, decide:

1. **Which terms to add?** (see Priority column)
2. **Guide storage location?** (classifier, separate DB, or both)
3. **Guide detail level?** (brief text vs step-by-step)
4. **Trigger mechanism?** (automatic in template vs on-demand via assistant)
5. **Should guides include images/videos?** (future enhancement)

---

## Next Steps

1. Review this list and select terms to implement
2. Choose implementation approach
3. Create technique guide database structure
4. Integrate with template generator and/or conversational assistant
5. Test with sample procedures

---

*Document created: 2025-11-19*
*Last updated: 2025-11-19*
