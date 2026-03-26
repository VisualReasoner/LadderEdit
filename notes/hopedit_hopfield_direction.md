# HopEdit Hopfield Direction

## Why the current patch is Hopfield-grounded

The current HopEdit bottleneck is no longer raw retrieval. On the mined collision subset, dual-whitened routing improves substantially over semantic-only, but hard-case post-edit accuracy stays flat. That means the next practical gain must come from increasing basin separation at the edit-cell level, not just improving top-k retrieval.

The collision-aware patch implements that idea directly:
- multi-view positive training on rewrite and rephrase forms strengthens the intended attractor basin for one edit
- sibling-negative suppression penalizes the current edit adapter if it assigns high likelihood to its own target on a high-conflict neighbor prompt
- this is an energy-margin style update: the intended basin should be low-energy on its own prompt family and higher-energy on nearby competing prompts

## Relevant Hopfield lines

### Classical Hopfield memory
- Hopfield 1982 framed associative memory as descent to stable energy minima and already highlighted content-addressable retrieval and spurious states.
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/

Implication for HopEdit:
- wrong-route events and abstentions on sibling prompts are the modern analogue of undesirable attractors or poorly separated minima
- reviewer-friendly framing: collision errors are associative-memory failures, not just retrieval-ranking mistakes

### Huge-capacity and dense associative memory
- Demircigil et al. studied associative memories with huge storage capacity.
- Link: https://ouci.dntb.gov.ua/en/works/4gRjJpE7/
- Krotov and Hopfield 2016 introduced dense associative memory and emphasized the feature-matching vs prototype regime.
- Link: https://papers.nips.cc/paper/6121-dense-associative-memory-for-pattern-recognition
- Krotov and Hopfield 2018 linked dense associative memory to robustness.
- Link: https://pubmed.ncbi.nlm.nih.gov/30314425/

Implication for HopEdit:
- our collision-heavy failures look prototype-like: semantically similar edits collapse toward the same basin
- whitening plus basin-separation loss should move the router-plus-cell system toward a more feature-sensitive regime

### Modern Hopfield networks and attention
- Ramsauer et al. 2020 showed attention is the update rule of a modern Hopfield network with continuous states.
- Link: https://huggingface.co/papers/2008.02217

Key points relevant to HopEdit:
- one-step retrieval is natural
- storage can scale exponentially with dimension under the modern Hopfield view
- there are different fixed-point types, including metastable subset states

Implication for HopEdit:
- collision subsets are exactly where metastable subset states are dangerous: the system averages over a sibling cluster rather than isolating one edit
- sparse routing plus confidence gating reduces unwanted averaging
- collision-aware edit training increases the margin between neighboring basins after routing

## What is now implemented

The current patch adds:
- multi-view positive edit training using rewrite, rephrase, and subject-conditioned prompts
- hard-negative mining from the current memory using the same dual-key conflict geometry as routing
- a basin-separation loss that penalizes the new adapter when it prefers its own target on high-conflict neighbor prompts
- conflict-neighbor export in memory snapshots for later theory analysis

## What to test next

### Practical-improvement tests for HopEdit
- rerun the mined collision subset with the new collision-aware config
- compare against the previous dual-whitened run on:
  - post rewrite accuracy
  - post rephrase accuracy
  - rewrite and rephrase route accuracy
  - abstain count vs wrong-route count
- if hard-case task accuracy rises while locality stays intact, the method story becomes much stronger

### Theory-facing tests for Edit Capacity
- test whether max combined conflict predicts abstention separately from wrong-route errors
- test whether the new basin-separation loss reduces same-subject confusion more than it reduces generic coverage
- report conflict-conditioned buckets rather than a single correlation coefficient

## If the new patch still fails

The next NeurIPS-level step should be a learned key projection head, not more blind LoRA tuning.

Reason:
- if edit-cell strengthening still does not convert routing gains into task gains, then the geometry itself is not expressive enough
- at that point the right method move is to learn the associative space, not just regularize the edit modules
