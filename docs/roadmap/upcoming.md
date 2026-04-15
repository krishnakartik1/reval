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

## Publish-results button

**Status: design sketch, deferred to webui scaffolding.**

When `reval-webui` is built, it needs a "Publish results"
affordance that takes a completed reval run (`results.json` +
per-run HTML report) and pushes it to the public leaderboard
pipeline. Open design questions:

- One-click GitHub Action trigger?
- Direct write to the leaderboard data directory?
- Manifest-signing step that the deploy job picks up?

Capturing the requirement now so it lands in the initial webui
scaffold rather than as an after-the-fact bolt-on.

## PyPI publication + version pinning

**Status: deferred ~4 weeks from docs-tab planning (revisit mid-2026-05).**

Today `reval-collector` depends on reval via an editable install:

```toml
# reval-collector/pyproject.toml
[tool.uv.sources]
reval = { path = "../reval", editable = true }
```

That works because the two sub-repos are siblings inside
`~/Documents/reval-workspace/`. Once reval hits a stable minor
version, the plan is:

1. Publish reval to PyPI (or at least a git tag).
2. Switch collector's `tool.uv.sources` to a semver range like
   `reval>=0.3,<0.4`.
3. Adopt a `CHANGELOG.md [CONTRACT]` section for breaking changes,
   so downstream consumers (collector, webui) can pin against a
   known-good minor version.

This unblocks distributing reval independently of the sibling-
repo layout.

## Release coordinator agent

**Status: deferred until version pinning.**

Once reval is published and collector pins against a release, the
plan is to add a `release-coordinator` agent to the workspace-level
`.claude/agents/` directory. It would handle:

- Bumping reval's version.
- Regenerating the CHANGELOG for any `[CONTRACT]` breakage.
- Opening a coordinated PR in collector (and eventually webui)
  to bump the pinned range.

For now, the existing `reval-architect`, `contract-impact`, and
`cross-repo-pr-reviewer` agents are sufficient — they catch
breakage, they just don't automate the version bump.

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

## Drift tests

**Status: open follow-up.**

Several docs pages duplicate prose from `reval/README.md` and the
rubric YAML files (install steps, scoring formulas, rubric
criteria). This was a deliberate v1 tradeoff — building a full
README-to-docs extractor would have tripled the docs tab scope.

The follow-up is a lightweight grep-based drift test: assert that
specific substrings ("`pip install -e`", "`0.85`", each rubric
criterion name) appear in both the README and the corresponding
docs page. If they drift apart, CI fails and you know to update
both.

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
