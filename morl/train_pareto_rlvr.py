#!/usr/bin/env python3
"""Pareto-style RLVR for LLMs (Hugging Face + PyTorch).

This script treats each prompt as a contextual bandit over sampled completions.
It supports three update modes:

- frontier: dominance-depth advantage from Pareto front membership.
- hv: hypervolume-contribution advantage (Pareto indicator based).
- mgda: vector advantage + multi-objective gradient combination.

It also includes a set-valued Pareto Bellman demo to make the operator explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(__file__))

from pareto_ops import (
    approx_hypervolume_contributions,
    bellman_union_over_actions,
    frontier_advantages,
    non_dominated_indices,
)


Vector = Tuple[float, ...]

UNSAFE_TERMS = {
    "kill",
    "bomb",
    "poison",
    "explosive",
    "harm",
    "suicide",
    "terror",
}


@dataclass
class CompletionSample:
    prompt_id: int
    approach_id: int
    prompt: str
    completion: str
    reward: Vector
    logprob: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pareto RLVR training loop")
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_prompts", type=int, default=2)
    parser.add_argument("--completions_per_prompt", type=int, default=4)
    parser.add_argument("--num_approaches", type=int, default=1)
    parser.add_argument("--completions_per_approach", type=int, default=None)
    parser.add_argument("--approach_token_prefix", type=str, default="APPROACH")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mode", type=str, default="frontier", choices=["frontier", "hv", "mgda"])
    parser.add_argument("--epsilon_dominance", type=float, default=0.0)
    parser.add_argument("--hv_samples", type=int, default=2000)
    parser.add_argument("--coverage_size", type=int, default=2)
    parser.add_argument("--coverage_diversity", type=float, default=0.6)
    parser.add_argument("--dup_penalty", type=float, default=0.35)
    parser.add_argument(
        "--selector_weights",
        type=str,
        default=None,
        help="Optional inference-time chooser weights over objectives, e.g. 0.5,0.3,0.2",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--show_bellman_demo", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_prompts() -> List[str]:
    return [
        "Explain the trade-off between speed and safety in autonomous driving in 4 sentences.",
        "Write a concise answer: Why is test-time verification useful for LLMs?",
        "Provide a short plan to debug a failing CI pipeline.",
        "Describe two risks of scalarizing multi-objective rewards too early.",
        "Give a practical example of Pareto non-dominance in model outputs.",
    ]


def load_prompts(path: str | None) -> List[str]:
    if path is None:
        return default_prompts()

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                rows.append(item)
    if not rows:
        raise ValueError("prompts file is empty")
    return rows


def _tokenize_words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", text)]


def _prompt_keywords(prompt: str) -> List[str]:
    words = _tokenize_words(prompt)
    return [w for w in words if len(w) >= 5]


def score_reward_vector(prompt: str, completion: str, target_len: int = 48) -> Vector:
    """Heuristic 3D reward vector.

    Replace this function with your rubric/evaluator stack:
      [task_quality, safety, style]
    """
    completion_words = _tokenize_words(completion)

    # Objective 1: task relevance proxy (keyword coverage).
    kws = set(_prompt_keywords(prompt))
    if kws:
        covered = len([w for w in kws if w in completion_words])
        task_quality = covered / float(len(kws))
    else:
        task_quality = 0.0

    # Objective 2: safety proxy (penalize unsafe terms).
    unsafe_hits = sum(1 for w in completion_words if w in UNSAFE_TERMS)
    safety = max(0.0, 1.0 - 0.5 * unsafe_hits)

    # Objective 3: style/conciseness proxy (target output length and sentence ending).
    n = len(completion_words)
    conciseness = max(0.0, 1.0 - abs(n - target_len) / float(target_len))
    formatting = 1.0 if completion.strip().endswith((".", "!", "?")) else 0.6
    style = 0.7 * conciseness + 0.3 * formatting

    return (float(task_quality), float(safety), float(style))


def parse_selector_weights(text: str | None) -> List[float] | None:
    if text is None:
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        return None
    s = sum(vals)
    if s == 0:
        raise ValueError("selector weights must not sum to 0")
    return [v / s for v in vals]


def approach_token(prefix: str, approach_id: int) -> str:
    return f"[{prefix}_{approach_id}]"


def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(_tokenize_words(a))
    sb = set(_tokenize_words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _pair_distance(i: int, j: int, rewards: Sequence[Vector], texts: Sequence[str], text_weight: float) -> float:
    ri, rj = rewards[i], rewards[j]
    rdist = sum((a - b) ** 2 for a, b in zip(ri, rj)) ** 0.5
    tdist = 1.0 - _jaccard_similarity(texts[i], texts[j])
    return rdist + text_weight * tdist


def _select_diverse_subset(
    candidate_indices: Sequence[int],
    rewards: Sequence[Vector],
    texts: Sequence[str],
    max_keep: int,
    text_weight: float,
) -> List[int]:
    """Greedy farthest-point subset to preserve coverage across Pareto survivors."""
    if max_keep <= 0:
        return []
    if len(candidate_indices) <= max_keep:
        return list(candidate_indices)

    # Seed with point farthest from survivor centroid.
    center = [
        sum(rewards[i][d] for i in candidate_indices) / float(len(candidate_indices))
        for d in range(len(rewards[0]))
    ]
    first = max(
        candidate_indices,
        key=lambda i: sum((rewards[i][d] - center[d]) ** 2 for d in range(len(center))),
    )
    chosen = [first]
    remaining = [i for i in candidate_indices if i != first]

    while remaining and len(chosen) < max_keep:
        nxt = max(
            remaining,
            key=lambda i: min(
                _pair_distance(i, j, rewards=rewards, texts=texts, text_weight=text_weight) for j in chosen
            ),
        )
        chosen.append(nxt)
        remaining.remove(nxt)

    return chosen


def _duplicate_penalties(approach_ids: Sequence[int], texts: Sequence[str]) -> List[float]:
    out = [0.0] * len(texts)
    for i in range(len(texts)):
        sims = [
            _jaccard_similarity(texts[i], texts[j])
            for j in range(len(texts))
            if i != j and approach_ids[i] == approach_ids[j]
        ]
        out[i] = max(sims) if sims else 0.0
    return out


def choose_from_frontier(
    rewards: Sequence[Vector],
    nd_indices: Sequence[int],
    selector_weights: Sequence[float] | None,
) -> int | None:
    """Explicit chooser: pick one completion from current frontier."""
    if not nd_indices:
        return None
    if selector_weights is None:
        return nd_indices[0]
    if len(selector_weights) != len(rewards[0]):
        raise ValueError("selector_weights dimension mismatch")

    return max(
        nd_indices,
        key=lambda i: sum(w * x for w, x in zip(selector_weights, rewards[i])),
    )


def sample_sequences(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        sequences = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return sequences, prompt_len


def continuation_logprobs(
    model: AutoModelForCausalLM,
    sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum log-probs of completion tokens only (per sequence)."""
    logits = model(sequences).logits[:, :-1, :]
    next_tokens = sequences[:, 1:]
    tok_logp = F.log_softmax(logits, dim=-1).gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

    # logits[t] predicts token t+1, so completion tokens start at index prompt_len-1.
    return tok_logp[:, prompt_len - 1 :].sum(dim=1)


def center_rewards_by_prompt(rewards: torch.Tensor, prompt_ids: Sequence[int]) -> torch.Tensor:
    out = rewards.clone()
    ids = torch.tensor(prompt_ids, device=rewards.device)
    for pid in ids.unique().tolist():
        mask = ids == int(pid)
        out[mask] = out[mask] - out[mask].mean(dim=0, keepdim=True)
    return out


def flatten_grads(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(g.reshape(-1))
    return torch.cat(chunks)


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project vector to probability simplex {x>=0, sum x = 1}."""
    if v.numel() == 1:
        return torch.ones_like(v)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / idx > 0
    nz = torch.nonzero(cond, as_tuple=False)
    if nz.numel() == 0:
        return torch.full_like(v, 1.0 / v.numel())
    rho = int(nz[-1, 0].item())
    theta = cssv[rho] / float(rho + 1)
    return torch.clamp(v - theta, min=0.0)


def mgda_weights(grad_matrix: torch.Tensor, steps: int = 80, lr: float = 0.2) -> torch.Tensor:
    """Find simplex weights minimizing ||sum_i alpha_i g_i||^2."""
    m = grad_matrix.shape[0]
    if m == 1:
        return torch.ones(1, device=grad_matrix.device, dtype=grad_matrix.dtype)

    gram = grad_matrix @ grad_matrix.T
    alpha = torch.full((m,), 1.0 / m, device=grad_matrix.device, dtype=grad_matrix.dtype)

    for _ in range(steps):
        grad = gram @ alpha
        alpha = project_to_simplex(alpha - lr * grad)

    return alpha


def compute_frontier_loss(
    samples: Sequence[CompletionSample],
    mode: str,
    epsilon_dominance: float,
    hv_samples: int,
    coverage_size: int,
    coverage_diversity: float,
    dup_penalty: float,
    selector_weights: Sequence[float] | None,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    logprobs = torch.stack([s.logprob for s in samples])
    rewards = [s.reward for s in samples]
    prompt_ids = [s.prompt_id for s in samples]
    approach_ids = [s.approach_id for s in samples]
    texts = [s.completion for s in samples]

    weights = [0.0] * len(samples)
    total_frontier = 0
    total_selected = 0
    selector_events: List[dict] = []

    unique_ids = sorted(set(prompt_ids))
    for pid in unique_ids:
        idx = [i for i, x in enumerate(prompt_ids) if x == pid]
        group = [rewards[i] for i in idx]
        group_texts = [texts[i] for i in idx]
        group_approaches = [approach_ids[i] for i in idx]

        nd = non_dominated_indices(group, epsilon=epsilon_dominance)
        nd_set = set(nd)
        total_frontier += len(nd)
        selected = _select_diverse_subset(
            candidate_indices=nd,
            rewards=group,
            texts=group_texts,
            max_keep=min(coverage_size, len(nd)) if coverage_size > 0 else len(nd),
            text_weight=coverage_diversity,
        )
        selected_set = set(selected)
        total_selected += len(selected_set)

        if mode == "hv":
            dim = len(group[0])
            reference = tuple(min(v[d] for v in group) - 1.0 for d in range(dim))
            contrib = approx_hypervolume_contributions(group, reference=reference, samples=hv_samples, seed=pid + 17)
            max_abs = max(abs(c) for c in contrib) if contrib else 0.0
            group_w = [(c / max_abs) if max_abs > 0 else 0.0 for c in contrib]
        else:
            group_w = frontier_advantages(group, epsilon=epsilon_dominance)

        # Keep all non-dominated points alive, but upweight a diverse coverage subset.
        for local_i in range(len(group_w)):
            if local_i in selected_set:
                group_w[local_i] = max(group_w[local_i], 1.0)
            elif local_i in nd_set:
                group_w[local_i] = max(group_w[local_i], 0.2)
            else:
                group_w[local_i] = min(group_w[local_i], -0.2)

        # Penalize duplicates within the same approach latent z.
        dup = _duplicate_penalties(group_approaches, group_texts)
        group_w = [w - dup_penalty * d for w, d in zip(group_w, dup)]

        chosen = choose_from_frontier(group, nd_indices=selected, selector_weights=selector_weights)
        if chosen is not None:
            selector_events.append({"prompt_id": int(pid), "chosen_approach": int(group_approaches[chosen])})

        for local, global_idx in enumerate(idx):
            weights[global_idx] = group_w[local]

    w = torch.tensor(weights, device=device, dtype=logprobs.dtype)
    loss = -(w.detach() * logprobs).mean()
    stats = {
        "frontier_rate": total_frontier / float(len(samples)),
        "selected_rate": total_selected / float(len(samples)),
        "avg_weight": float(w.mean().item()),
        "selector": selector_events,
    }
    return loss, stats


def compute_mgda_loss(
    samples: Sequence[CompletionSample],
    params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    logprobs = torch.stack([s.logprob for s in samples])
    rewards = torch.tensor([s.reward for s in samples], device=device, dtype=logprobs.dtype)
    prompt_ids = [s.prompt_id for s in samples]

    centered = center_rewards_by_prompt(rewards, prompt_ids)

    objective_losses: List[torch.Tensor] = []
    grad_rows: List[torch.Tensor] = []
    for i in range(centered.shape[1]):
        li = -(centered[:, i].detach() * logprobs).mean()
        objective_losses.append(li)
        grads = torch.autograd.grad(li, params, retain_graph=True, allow_unused=True)
        grad_rows.append(flatten_grads(grads, params))

    grad_matrix = torch.stack(grad_rows)
    alpha = mgda_weights(grad_matrix.detach())
    losses = torch.stack(objective_losses)
    combined_loss = torch.sum(alpha * losses)

    stats = {
        "alpha": [round(float(x), 4) for x in alpha.detach().cpu().tolist()],
        "reward_mean": [round(float(x), 4) for x in rewards.mean(dim=0).detach().cpu().tolist()],
    }
    return combined_loss, alpha, stats


def run_bellman_demo() -> None:
    print("\n=== Set-Valued Pareto Bellman Demo ===")
    rewards_by_action = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    next_frontiers_by_action = [
        [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0)],
        [(1.0, 0.0, 1.0), (0.0, 2.0, 0.0)],
    ]
    gamma = 0.95

    v_star = bellman_union_over_actions(
        rewards_by_action=rewards_by_action,
        next_frontiers_by_action=next_frontiers_by_action,
        gamma=gamma,
        epsilon=0.0,
        max_points=None,
    )

    print("V*(s) = ND( union_a [ R(s,a) + gamma * V(s') ] )")
    print("Result frontier:")
    for v in v_star:
        print(f"  {v}")


def main() -> None:
    args = parse_args()
    if args.num_approaches < 1:
        raise ValueError("--num_approaches must be >= 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    selector_weights = parse_selector_weights(args.selector_weights)

    prompts = load_prompts(args.prompts_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if args.show_bellman_demo:
        run_bellman_demo()

    if args.completions_per_approach is not None and args.completions_per_approach < 1:
        raise ValueError("--completions_per_approach must be >= 1 when provided")
    if args.completions_per_approach is None:
        if args.num_approaches == 1:
            completions_per_approach = args.completions_per_prompt
        else:
            completions_per_approach = max(1, args.completions_per_prompt // args.num_approaches)
    else:
        completions_per_approach = args.completions_per_approach

    for step in range(1, args.steps + 1):
        chosen_prompts = [random.choice(prompts) for _ in range(args.batch_prompts)]

        samples: List[CompletionSample] = []
        for pid, prompt in enumerate(chosen_prompts):
            for approach_id in range(args.num_approaches):
                conditioned_prompt = f"{approach_token(args.approach_token_prefix, approach_id)} {prompt}"
                sequences, prompt_len = sample_sequences(
                    model,
                    tokenizer,
                    prompt=conditioned_prompt,
                    num_sequences=completions_per_approach,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )

                seq_logprobs = continuation_logprobs(model, sequences, prompt_len)
                continuation_tokens = sequences[:, prompt_len:]
                completions = tokenizer.batch_decode(continuation_tokens, skip_special_tokens=True)

                for i, text in enumerate(completions):
                    reward = score_reward_vector(prompt, text)
                    samples.append(
                        CompletionSample(
                            prompt_id=pid,
                            approach_id=approach_id,
                            prompt=prompt,
                            completion=text,
                            reward=reward,
                            logprob=seq_logprobs[i],
                        )
                    )

        optimizer.zero_grad(set_to_none=True)

        if args.mode in {"frontier", "hv"}:
            loss, stats = compute_frontier_loss(
                samples=samples,
                mode=args.mode,
                epsilon_dominance=args.epsilon_dominance,
                hv_samples=args.hv_samples,
                coverage_size=args.coverage_size,
                coverage_diversity=args.coverage_diversity,
                dup_penalty=args.dup_penalty,
                selector_weights=selector_weights,
                device=device,
            )
            loss.backward()
        else:
            loss, _, stats = compute_mgda_loss(samples=samples, params=trainable_params, device=device)
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
        optimizer.step()

        reward_means = torch.tensor([s.reward for s in samples], dtype=torch.float32).mean(dim=0).tolist()
        payload = {
            "step": step,
            "mode": args.mode,
            "loss": round(float(loss.item()), 5),
            "grad_norm": round(float(grad_norm.item()), 5),
            "reward_mean": [round(float(x), 4) for x in reward_means],
        }
        payload.update(stats)
        print(json.dumps(payload))

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"saved model to {out}")


if __name__ == "__main__":
    main()
