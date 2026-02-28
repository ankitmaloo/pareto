# Notes: Pareto Reward Vectors for LLMs

## Core idea
Scalar reward compresses many objective-level judgments into one number. That destroys structure:
- Binary reward gives very little information per sample.
- Averaging rubric scores forces trade-offs to be decided early.
- Two responses can be qualitatively different but both strong, yet scalarization picks one winner.

Pareto framing keeps reward as a vector and uses dominance:
- `a` dominates `b` if `a` is at least as good in all objectives and better in at least one.
- If neither dominates, they are **incomparable** (both can be treated as valid directions).

This avoids over-penalizing alternative good solutions.

## Information perspective (intuition)
If we collapse to scalar buckets, the label space is limited by scalar outcomes.
With vector rewards, each dimension contributes structure to pairwise relations:
- dominate / dominated / incomparable.
- This can expose richer learning signal, especially when objectives conflict.

## What the new script does
`pareto_advantage_demo.py` includes:
- Pareto dominance checks.
- Pareto front extraction.
- Scalar advantage (baseline: mean scalar score).
- Pareto-aware advantage surrogate using pairwise relations:
  - `+1` for dominating another response
  - `-1` for being dominated
  - `0` for incomparable

It includes a curated 3D example matching your intuition:
- `[1,1,1]` is dominated by vectors like `[1,1,2]` and `[1,2,1]`.
- `[1,1,2]` and `[1,2,1]` are incomparable.

## Why this matters for LLM training
In preference optimization / RLHF-like loops, scalar rewards force total ordering.
Pareto-style rewards allow partial ordering:
- stronger against clearly dominated outputs,
- neutral among diverse-but-strong outputs.

This can help preserve response diversity while still improving quality.

## Next experiments I would run
1. Replace pairwise binary preference with ternary labels: win / loss / incomparable.
2. Train with two heads: dominance classifier + scalar fallback head.
3. Compare collapse/diversity metrics against scalar baselines.
4. Test adaptive scalarization only at sampling time (not training time).

## Alternative variant (explicit z, kept for reference)
Single-policy RL can still collapse even with vector rewards unless we explicitly preserve modes.

Practical fix:
1. Add a latent approach index `z` (input token like `[APPROACH_2]`).
2. Sample multiple completions per prompt across multiple `z`.
3. Keep non-dominated candidates; then keep a diverse subset of survivors.
4. Penalize duplicates within the same `z`.
5. Train on survivor-conditioned log-prob updates (not one global winner).

This turns the model into a controllable menu:
- model learns multiple valid solution styles,
- user/system can select one style later using dynamic preference weights.

The new `morl/train_pareto_rlvr.py` now implements this pattern in `frontier`/`hv` modes.

## Current direction (Variant B): credit assignment to avoid collisions
Current hypothesis direction is no explicit mode token/adapters.
The main risk is gradient collision when two incomparable survivors share a prefix and diverge later.

My view: credit assignment is the cleanest fix.
- The collision is usually localized at branch regions, not the whole sequence.
- Shared regions should be reinforced normally.
- Branch regions should preserve optionality (targeted entropy, not blanket entropy).
- Post-branch regions should be scored conditionally by chosen approach.

Practical recipe:
1. Sample N candidates, score vector rewards, keep Pareto survivors.
2. Detect shared vs branch vs post-branch regions (token diff + embedding divergence).
3. Apply region-specific advantages/penalties instead of one sequence-level reward.
4. Use branch-conditional baselines and deduplicate near-identical survivors.

This keeps Pareto diversity without forcing outputs into a blended compromise.
Scratchpad with the full collision discussion: [scratchpad/credit_assignment_collision.md](./scratchpad/credit_assignment_collision.md)
