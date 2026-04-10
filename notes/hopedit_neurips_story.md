# HopEdit NeurIPS Story

## Main Contribution Order

1. **Theory**
   - Lifelong editing breaks through three coupled failure modes:
     - access / retrieval failure
     - realization / update failure
     - locality / interference failure
   - Collision structure, not edit count alone, governs practical capacity.

2. **Measurement**
   - Use a unified package of:
     - rewrite / rephrase / locality
     - distortion
     - failure decomposition
     - route diagnostics
     - efficiency metrics
   - Keep proxy labels explicit for non-routed methods.

3. **Method**
   - HopEdit is motivated by this theory.
   - It aims to improve separability through:
     - calibrated feature geometry
     - routed access
     - collision-aware realization

## Positioning Against Closest Papers

### Knowledge in Superposition

- Their strength: strong interference diagnosis.
- Our differentiation:
  - move from overwrite/interference-only diagnosis to a fuller access + realization + locality account
  - make the failure structure actionable through method design and measurement

### MEMOIR / WikiBigEdit-style lifelong methods

- Their strength: strong lifelong engineering systems.
- Our differentiation:
  - explain *why* failure occurs
  - expose *which* failure mode dominates
  - derive HopEdit from the resulting theory

## Main Experimental Spine

1. **Controlled diagnosis**
   - collision-heavy streams vs cleaner streams
   - conflict beats count / position

2. **Real lifelong validation**
   - official WikiBigEdit increment protocol
   - matched checkpoints
   - longitudinal trajectories

3. **Realistic evaluation**
   - non-teacher-forcing / WILD-style evaluation in the main story

## Claim Discipline

- Do not treat HopEdit as the theory itself.
- Do not compare unmatched checkpoints in main tables.
- Do not overclaim downstream robustness beyond directly evaluated scales.
- If a baseline is unstable or unfair, move it out of the main comparison or label it explicitly.
