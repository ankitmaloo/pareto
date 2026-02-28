# MORL (Pareto RLVR for LLMs)

This folder contains a practical starter for **multi-objective RLVR** with vector rewards.

## Basic steps (first formulation)
1. Define objective-wise reward vector `r = [r1, r2, ..., rm]` per completion (do not average rubric scores).
2. Sample completions per prompt across latent approach IDs `z` (menu inside one model).
3. Score each completion into a reward vector.
4. Compute Pareto signal per prompt:
   - `frontier` mode: non-dominated/dominated signal via dominance depth.
   - `hv` mode: hypervolume contribution signal.
   - `mgda` mode: vector advantages per objective + multi-gradient common descent direction.
5. Keep a coverage subset of Pareto survivors (quality + spread), not a single winner.
6. Apply anti-duplication penalty within each `z` to reduce mode collapse.
7. Update policy with weighted log-prob objective conditioned on `z`.
8. Optionally choose one candidate at inference via explicit selector weights.
9. Keep pruning/approximating (epsilon-dominance, capped archive/frontier) to avoid set explosion.
10. Choose trade-off policy late (inference-time preference), not early (training-time fixed scalar weights).

## Files
- `pareto_ops.py`: Pareto dominance, frontier pruning, set-valued Bellman backup, hypervolume contribution approximation.
- `train_pareto_rlvr.py`: Hugging Face + PyTorch training loop in RLVR contextual-bandit style with latent approach token `z`.
- `CODE_WALKTHROUGH.md`: function-by-function walkthrough of training flow and design choices.

## Important note
`train_pareto_rlvr.py` contains a **heuristic placeholder scorer** (`score_reward_vector`) so the code runs end-to-end.
Replace it with your real verifier/rubric/reward-model evaluators.

## Quick start (remote env)
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers
python morl/train_pareto_rlvr.py --mode frontier --steps 5 --num_approaches 4 --completions_per_approach 2 --coverage_size 3 --dup_penalty 0.35 --show_bellman_demo
python morl/train_pareto_rlvr.py --mode mgda --steps 5
python morl/train_pareto_rlvr.py --mode hv --steps 5 --hv_samples 1000
```

## Latent approach control
- `--num_approaches`: number of latent approach IDs (`z` values).
- `--completions_per_approach`: samples per prompt per `z`.
- `--approach_token_prefix`: text prefix used to create conditioning token (example `"[APPROACH_2]"`).
- `--coverage_size`: number of Pareto survivors kept with diversity-aware coverage per prompt.
- `--coverage_diversity`: relative weight of textual diversity in coverage selection.
- `--dup_penalty`: similarity penalty among outputs from the same `z`.
- `--selector_weights`: optional inference-time chooser over reward objectives (example `0.6,0.3,0.1`).

## Prompt file format
Pass `--prompts_file path.txt` where each non-empty line is one prompt.

## Bellman view (set-valued)
The script includes a demo of:

`V*(s) = ND( union_a [ R(s,a) + gamma * V(s') ] )`

where `ND` keeps non-dominated vectors.
