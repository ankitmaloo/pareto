# Hypothesis

**Problem (with an example)**  
LLM training feedback is usually collapsed to a single scalar ("overall score"). That forces a *total order* even when answers are incomparable, so training picks one “winner” and squeezes out alternative valid approaches.

Example: two answers to the same prompt

- A: **very correct**, dry/terse
- B: **slightly less precise**, very clear/teachable

If you average rubrics, you’ll often declare one “better overall,” even though they’re both reasonable depending on what the user wants.

**Hypothesis**  
If rewards stay as a **vector** (one dimension per objective) and we respect the **partial order** (Pareto dominance + “incomparable”), the model can keep multiple “right” modes alive instead of collapsing into one compromise.

Example reward vectors (higher is better):

- A = [correctness 10, clarity 6, creativity 2]
- B = [correctness 8, clarity 10, creativity 2]

Neither dominates the other -> they’re both valid.

**Proposed solution**  
**Pareto-aware RLVR** (don’t crown a single winner unless dominance is clear):

1. Generate `k` diverse completions.
2. Score each completion with a reward vector (correctness, safety, clarity, concision, etc.).
3. Keep the **non-dominated set** (Pareto front).
4. If X dominates Y (`>=` on all, `>` on at least one), reinforce X and penalize Y.
5. If X and Y are incomparable, reinforce both (or keep both in the accepted set).
6. Add explicit **anti-duplication** pressure so the accepted set stays meaningfully diverse (not 5 paraphrases of the same approach).
7. Add **adaptive exploration pressure** so the model branches more on hard/uncertain prompts and less on easy prompts.

This variant does **not** require explicit approach tokens or adapters; diversity is encouraged through frontier coverage + non-redundancy.

**Outcome I want (with an example)**  
A model that behaves like a good consultant: it can hold multiple valid approaches and select based on context.

Think of it like this: you don’t actually want “a mode knob.” You want **option value**.

Humans don’t keep a labeled library of modes in their head; they do a fast loop:

1. Generate a few plausible approaches.
2. Sanity-check them.
3. Commit to one (or ask a clarifying question).
4. Keep the ability to pivot.

So the right framing isn’t “train modes.” It’s **train a policy that (a) can generate diverse candidate thoughts when it’s useful and (b) can select a good one.** The diversity is *latent*—you don’t have to name it.

## What to optimize (instead of a single answer)

Stop treating “the output” as the thing you optimize. Treat the model as producing a **set of candidates** internally.

Objective becomes something like:

- Maximize expected quality of the best candidate (best-of-N / self-consistency), plus
- Maximize diversity among candidates (avoid N paraphrases), plus
- Minimize waste (don’t explore when the answer is obvious).

That’s the “exploration on the fly” behavior you want.

## How to train it (RLVR-compatible, no explicit modes)

For each prompt during training:

1. Sample N completions (temperature > 0, varied decoding).
2. Score each completion with your reward vector (verifiers + rubrics).
3. Keep the **non-dominated** ones (Pareto survivors).
4. Update the model to increase probability of survivors, decrease dominated ones.
5. Add a **novelty / anti-duplication term** so survivors are meaningfully different (different plan structure, different key assumptions, different proof route—not just synonyms).
6. Add **adaptive exploration pressure**: preserve entropy more on hard/uncertain prompts, less on easy ones.

No explicit “mode” needed. Diversity emerges because you’re rewarding frontier coverage + non-redundancy.

## How to run it at inference (model decides on the fly)

You can make exploration conditional without exposing a knob:

- The model first estimates “am I uncertain / is this multi-solution?” (implicit or explicit).
- If yes: it does a small internal best-of-N / branch-and-choose.
- If no: it answers directly.

That’s basically test-time thinking with branching, but disciplined: branch only when it buys expected improvement.

## How to measure it (since diversity is squishy)

Measure the **value of sampling**:

- **Gain from N samples:** `performance(best-of-N) - performance(1-sample)`.  
  If that gap is real and consistent, you’ve trained genuine internal diversity that’s useful.
- **Redundancy:** average similarity among the N candidates (should be low *without* tanking correctness).
- **Frontier coverage:** size/spread of the Pareto survivor set per prompt (or hypervolume contribution).
- **Pivot behavior:** on prompts with hidden pitfalls, does the model produce at least one candidate that avoids the trap?

If you get “best-of-N improves a lot on hard prompts, and candidates aren’t redundant,” you’ve basically achieved more exploration in a measurable way.

Net: you’re not building a model with modes. You’re building **a model with a healthy internal search process**—one that generates a small portfolio of plausible solutions, then picks, like a competent human who’s had coffee.
