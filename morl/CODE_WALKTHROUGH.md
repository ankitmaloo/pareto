# Code Walkthrough: Pareto RLVR Trainer

This walkthrough explains how `/Users/ankit/Documents/dev/RL/pareto/morl/train_pareto_rlvr.py` works end-to-end.

## 1) High-level objective
Train an LLM with **vector rewards** (not pre-averaged scalar reward), while:
- keeping multiple valid solution modes alive,
- avoiding collapse to one “average best” style,
- enabling inference-time preference shifts.

The script does this with:
- Pareto survivor selection,
- latent approach token `z`,
- duplicate penalty within each `z`,
- optional explicit selector weights at inference-time.

## 2) Main data structure
`CompletionSample` stores one sampled completion:
- `prompt_id`: which prompt in current minibatch.
- `approach_id`: latent mode id (`z`).
- `prompt`: source prompt.
- `completion`: generated text.
- `reward`: reward vector `(r1, r2, ..., rm)`.
- `logprob`: sum of completion-token logprobs.

## 3) CLI configuration areas
Important flags in `parse_args()`:
- Model/training basics: `--model`, `--steps`, `--lr`, `--device`.
- Sampling: `--batch_prompts`, `--max_new_tokens`, `--temperature`, `--top_p`.
- Latent menu control: `--num_approaches`, `--completions_per_approach`, `--approach_token_prefix`.
- Pareto update mode: `--mode` (`frontier`, `hv`, `mgda`).
- Coverage/diversity: `--coverage_size`, `--coverage_diversity`, `--dup_penalty`.
- Chooser: `--selector_weights`.

## 4) Reward pipeline
`score_reward_vector(prompt, completion)` returns a 3D placeholder reward:
- task relevance proxy,
- safety proxy,
- style/conciseness proxy.

This is intentionally replaceable with your real evaluator stack.

## 5) Sampling with latent approach token `z`
In `main()` training loop:
1. Sample prompts.
2. For each prompt and each `approach_id`, prepend token text:
   - `"[APPROACH_<id>] <prompt>"`
3. Generate multiple completions.
4. Compute sequence logprobs with `continuation_logprobs(...)`.
5. Score each completion into reward vectors.

This gives per-prompt samples spread across latent mode IDs.

## 6) Pareto-style update path (`frontier` / `hv`)
`compute_frontier_loss(...)` does:
1. Group completions by prompt.
2. Compute non-dominated set (`non_dominated_indices` from `pareto_ops.py`).
3. Build weights:
   - `frontier` mode: dominance-depth-based weights.
   - `hv` mode: hypervolume contribution weights.
4. Keep coverage subset with `_select_diverse_subset(...)`:
   - preserves spread among survivors (reward-space + text-distance).
5. Apply anti-duplication with `_duplicate_penalties(...)`:
   - similar outputs under same `z` get penalized.
6. Optional chooser event:
   - `choose_from_frontier(...)` uses `--selector_weights` to pick one candidate from selected frontier points.
7. Final policy loss:
   - `loss = -mean(weight * logprob)`.

Net effect:
- non-dominated outputs are reinforced,
- dominated outputs are downweighted,
- repetitive same-`z` behavior is discouraged.

## 7) MGDA update path (`--mode mgda`)
`compute_mgda_loss(...)`:
1. Center vector rewards per prompt (`center_rewards_by_prompt`).
2. Build one objective loss per reward dimension.
3. Compute per-objective gradients.
4. Solve simplex weights with `mgda_weights(...)`:
   - find shared descent direction minimizing gradient conflict.
5. Use weighted sum of objective losses for one optimizer step.

This avoids scalarizing reward values directly while still producing one SGD update.

## 8) Set-valued Bellman operator support
`morl/pareto_ops.py` contains Pareto Bellman utilities:
- `dominates`, `non_dominated_indices`, `pareto_prune`.
- `pareto_bellman_backup(...)`: one-action set backup.
- `bellman_union_over_actions(...)`:
  - `V*(s) = ND( union_a [ R(s,a) + gamma * V(s') ] )`.

`--show_bellman_demo` prints a toy frontier using this operator.

## 9) Training step output
Each step prints JSON with key signals:
- `loss`, `grad_norm`,
- `reward_mean`,
- `frontier_rate`,
- `selected_rate`,
- `avg_weight`,
- and selector metadata when active.

Use these to monitor whether frontier coverage is being preserved vs collapsing.

## 10) What to replace first for production
1. Replace `score_reward_vector(...)` with real objective evaluators.
2. Replace Jaccard text similarity with embedding/semantic similarity.
3. Add explicit archive memory for frontier points across steps/prompts.
4. Add policy distillation path if you move to adapter population training.
