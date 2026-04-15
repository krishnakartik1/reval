---
title: Viewing reports
order: 3
description: Anatomy of the per-run HTML report and the static leaderboard site.
---

REVAL produces two kinds of HTML output:

1. **Per-run report** (`results/<run>/report.html`) — a
   self-contained dashboard for a single benchmark run.
2. **Static leaderboard site** (`public/`) — an aggregate view
   across every run in `showcase/`, plus this documentation tab.

Both share the same CSS tokens and brand theme defined in
`src/reval/leaderboard/assets/tokens.css`, so the leaderboard and
per-run reports look like one product.

## Per-run report anatomy

Each per-run `report.html` has four stacked sections:

### 1. Header

- Model handle and its resolved provider + model_id pair.
- Run timestamp, git SHA (`--dirty`-aware), judge model, embeddings
  model. This is the reproducibility stamp — runs with the same
  SHA against the same dataset should yield identical scores
  modulo judge nondeterminism.

### 2. Score summary

- Overall score (weighted mean across completed evals).
- One score chip per category, colored by the interpretation bands
  from [`evals/config.yaml`](../reference/config.md):
  - `≥ 0.85` — green (high)
  - `≥ 0.70` — yellow (medium)
  - `< 0.70` — red (potential bias)

### 3. Result cards

One expandable card per eval. Each card shows three sub-sections:

- **Test case** — the original prompt(s) and (where present) the
  counterfactual pair, ground truth, or figure pair from the dataset.
- **Model response** — what the target model said.
- **Scoring** — for rubric-scored categories (`figure_treatment`,
  `issue_framing`), a per-criterion breakdown with the judge's
  reasoning. For similarity-scored categories (`policy_attribution`,
  `factual_accuracy`), the raw similarity values and threshold.

### 4. Metadata footer

Provenance fields that don't fit in the header: full judge system
prompt, rubric YAML hash, environment, and the exact CLI invocation
that produced the run.

## Leaderboard site

`reval leaderboard build` aggregates every directory under
`showcase/` that contains a `results.json`. The output is a
self-contained `public/` tree:

```text
public/
├── index.html              # leaderboard table + sort/filter UI
├── models/
│   └── <slug>.html         # per-model detail page with radar chart
├── docs/                   # this docs tab (if docs are installed)
│   ├── index.html
│   └── <section>/<page>.html
├── reports/
│   └── <slug>.html         # per-run reports, regenerated against current dataset
├── data/
│   └── leaderboard.json    # raw rows for external consumers
└── assets/
    ├── tokens.css, style.css, docs.css, pygments.css
    └── radar.js
```

The leaderboard table is reactive — sorting and filtering happen
client-side via Alpine.js. No server is required; you can serve
`public/` from any static host.

## Previewing locally

```bash
# Build both leaderboard + docs
reval leaderboard build

# Serve
python -m http.server --directory public 8000
```

Common entry points:

- `http://localhost:8000/` — leaderboard index
- `http://localhost:8000/docs/` — this documentation tab
- `http://localhost:8000/models/<slug>.html` — individual model page

## Deploying

The leaderboard at [revalbench.com](https://revalbench.com) is deployed
via Cloudflare Pages, which auto-deploys on every push to `master`. No
CF config files are checked in; Cloudflare handles the full rebuild on
its side. If you're running your own deployment, `public/` is a plain
static directory — any host that can serve HTML + JSON will do.
