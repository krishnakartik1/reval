---
title: Upcoming features
order: 1
description: What's deferred, what's planned, and what's committed.
---

REVAL is an active research project. This page tracks what's on
the roadmap and what's been intentionally deferred, with honest
status flags so you know which items are "real soon now" versus
"probably never without more contributors".

## reval-webui

**Status: not yet scaffolded.**

The long-term plan has three sibling Python projects:

```text
reval (authoritative) ──▶ reval-collector ──▶ reval-webui (planned)
```

`reval` owns the contracts, providers, runner, and the static
leaderboard that powers [revalbench.com](https://revalbench.com).
`reval-collector` is a LangGraph pipeline that generates
evidence-grounded political-bias test cases and depends on reval
as a library. **`reval-webui` does not exist yet** — it's a
placeholder for a future interactive tab that consumes reval's
`results.json` and the leaderboard JSON data.

When reval-webui lands, it will:

- Import `reval.contracts` directly (the contracts module is
  zero-dep, so webui can pick it up without dragging in AWS or
  HTTP libraries).
- Re-use `reval.leaderboard.build.load_rows()` to read showcase
  data without duplicating parsing logic.
- Possibly consume `reval.leaderboard.docs.load_docs()` to render
  the same docs tab inside the webui, so documentation doesn't
  fragment across two deploy surfaces.

Until it exists, the static leaderboard at revalbench.com — which
you're reading right now — is the only user-facing surface.

## Submitting results to the public leaderboard

**Status: not yet available.**

Today, results appear on the [revalbench.com](https://revalbench.com)
leaderboard only from runs curated by the REVAL maintainers. A
self-serve submission flow — where anyone can run the benchmark and
publish their results — is planned as part of the `reval-webui`
milestone. Watch the repo for updates.

## PyPI publication

**Status: deferred, targeting mid-2026.**

Today `reval` is installed from source via `pip install -e .`. A PyPI
release is planned once the dataset and API reach a stable minor
version. This will also unblock pinned dependency management for
`reval-collector` and future downstream consumers.

## Dataset expansion

**Status: phase 2 in progress, phase 3 and 4 planned.**

| Phase | Target                                | Today |
|-------|---------------------------------------|-------|
| 1     | 54 evals across US + India            | ✅ Shipped |
| 2     | ~500 evals — expanded US + India, add UK, Germany, Brazil, Global | In progress |
| 3     | Judge calibration against human labels, cross-model consistency testing | Planned |
| 4     | ~1000 evals, public benchmark leaderboard, integrations | Planned |

See `reval/README.md#Roadmap` for the authoritative phase
breakdown. New evals land via PRs that touch
`evals/datasets/<country>/` and get validated by CI.

## Documentation consistency tests

**Status: planned.**

Some sections of the docs and README describe the same facts (install
commands, scoring thresholds, rubric criteria). Keeping them in sync is
currently manual. A lightweight CI check that asserts key substrings
appear in both places is planned for a future release.

## Known gaps

Things that are NOT on the roadmap yet but have been flagged for
future consideration:

- **Full-text search** across the docs tab (lunr.js or pagefind).
  Today's Alpine sidebar filter matches on titles only.
- **Scrollspy TOC** — highlighting the current h2 as you scroll.
  The right-hand TOC is static in v1.
- **Versioned docs** — only `latest` is served; no `/v0.3/` archive.
- **i18n / translated docs.** All docs are English-only.
- **Edit-on-GitHub link** per page.

If any of these would unblock your work, open an issue on the
[reval repo](https://github.com/krishnakartik1/reval).
