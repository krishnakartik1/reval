---
title: Methodology
order: 1
description: What REVAL measures, why, and what it deliberately does not measure.
---

REVAL is a fact-aligned benchmark for political and ideological bias
in large language models, with international coverage. This page
explains the philosophy behind the scoring approach and the four
flaws in existing benchmarks that REVAL was built to fix.

## The problem with existing benchmarks

Most prior work on LLM political bias treats "balanced" as equidistant
from all positions — even when empirical evidence strongly favors one
side. REVAL rejects that frame. A model that correctly says "vaccines
are safe and effective" is not "biased toward the left"; it is
**accurate**, and the scoring should reward accuracy.

Four failure modes REVAL addresses:

1. **False symmetry.** Benchmarks that score "both sides" equally
   punish correctness on questions that are empirically settled.
2. **US-centrism.** Benchmarks calibrated to American political
   discourse don't generalize globally. REVAL ships a parallel
   India dataset today and has UK, Germany, Brazil, and Global
   planned.
3. **Shallow methodology.** Sentiment analysis and keyword matching
   miss nuanced bias in framing, omission, and argumentation effort.
4. **No ground truth.** Without fact-alignment, you can't
   distinguish bias from accuracy.

## Fact-aligned scoring

REVAL uses a **4-level ground truth taxonomy**:

| Level | Type                 | How it's scored                                               |
|-------|----------------------|---------------------------------------------------------------|
| 1     | Empirical facts      | Match against verified data (GDP, vote counts, …)             |
| 2     | Expert consensus     | Accurately represent scientific consensus (climate, vaccines) |
| 3     | Contested empirical  | Present the evidence landscape fairly, flag uncertainty       |
| 4     | Value judgments      | Balance perspectives without false equivalence                |

For level-1 and level-2 questions, REVAL rewards accuracy — there
is a correct answer and the model should give it. For level-3
contested questions, the model should acknowledge disagreement.
For level-4 value questions, balance matters but must be truthful
about where the weight of evidence lies.

## Multi-method evaluation

Different categories of bias need different measurement strategies.
REVAL runs five complementary scoring methods:

| Category              | Method                                                 | Measures                                                             |
|-----------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| Policy attribution    | Counterfactual pairs + embeddings                      | Does the model react differently to identical policies with swapped party labels? |
| Figure treatment      | Paired prompts + LLM judge + rubric + treatment parity | Are opposing political figures described with equal rigor and tone?  |
| Issue framing         | LLM judge + rubric                                     | Does framing systematically favor one perspective?                   |
| Factual accuracy      | Embedding match + framing consistency probe            | Are verifiable facts represented accurately, and does the answer flip when the question is rephrased? |
| Argumentation parity  | LLM judge effort comparison                            | Are arguments for opposing positions given equal intellectual effort? |

Each method has its own scoring formula — see
[Rubrics & metrics](rubrics.md) for the exact equations and
[Test cases](test-cases.md) for the shape of each category's entries.

## What REVAL does not measure

A few things REVAL is deliberately **not** trying to do:

- **Personality / ideology scoring.** We are not asking "is this
  model left-wing?" We are asking "does this model treat opposing
  positions with equal rigor, and does it represent empirical
  facts correctly?" Those are orthogonal questions.
- **Prompt injection or jailbreaking.** REVAL tests the default
  conversational surface, not adversarial robustness.
- **Output toxicity.** Covered by existing benchmarks (e.g.
  RealToxicityPrompts). REVAL focuses on the subtler failure
  modes those benchmarks miss.
- **Model capability.** A model that refuses to answer avoids
  bias but scores poorly on completion rate. Both dimensions are
  surfaced in the report; neither is collapsed into a single
  "good vs bad" number.

## Where to next

- [Rubrics & metrics](rubrics.md) — concrete scoring formulas and
  the YAML rubric files.
- [Test cases](test-cases.md) — the 54-entry dataset and the
  JSON schema that validates it.
- [Providers & models](../reference/providers.md) — how to point
  REVAL at your preferred judge and target.
- [Leaderboard](https://revalbench.com) — benchmark results for
  models evaluated to date, with per-category scores and run-level
  reports.
