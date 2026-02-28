# Scratchpad: Gradient Collision and Credit Assignment

This scratchpad captures the collision discussion and the resulting direction for Variant B (no explicit mode token).

## Core trap
If a prompt has two valid, incomparable answers A and B, and training reinforces both at sequence level, gradients can interfere at shared prefixes.
This can produce blended outputs (mode averaging / "Frankenstein" outputs) instead of cleanly distinct solutions.

## Key observation
Collision is usually local, not global:
- shared prefix often aligns,
- divergence happens around a branch point,
- post-branch execution can be evaluated conditionally.

## Proposed resolution
Use surgical credit assignment instead of global sequence-level reward smearing:
1. Sample N candidates, score with vector rewards, keep Pareto survivors.
2. Align survivors to identify shared regions and branch regions.
3. Assign token/step-level credit:
   - Shared region: reinforce normally.
   - Branch region: keep entropy/optionality (avoid collapse).
   - Post-branch region: evaluate and reinforce conditionally by branch.
4. Localize penalties for dominated candidates to where they actually failed.
5. Deduplicate near-identical survivors before diversity/frontier updates.

## Why this matters
This keeps Variant B viable without explicit mode infrastructure.
The model can preserve on-the-fly search behavior by learning where to branch and where to execute deterministically.

## Practical caveats
- Branch-point detection is noisy and may be distributed over spans, not one token.
- Needs enough samples per prompt to estimate branch-conditioned baselines.
- Entropy at branch regions must be calibrated (too high hurts coherence, too low collapses diversity).

## Immediate implementation direction
- Add branch-region detection over candidate traces (token diff + embedding divergence).
- Replace sequence-level weights with per-token/per-span weights.
- Use branch-conditional baselines in advantage computation.
- Keep frontier coverage + anti-duplication in survivor selection.
