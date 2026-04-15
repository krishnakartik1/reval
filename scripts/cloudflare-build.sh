#!/usr/bin/env bash
# Cloudflare Pages build entrypoint for revalbench.com.
#
# Configure the CF dashboard (Pages → reval → Settings → Builds &
# deployments) with:
#
#   Build command:         bash scripts/cloudflare-build.sh
#   Build output directory: public
#
# Keeping the recipe in-repo means future changes to install extras,
# build flags, or the output contract land in version control with the
# rest of the code — no silent dashboard drift.
#
# Installs the `[docs]` extra so `reval leaderboard build` can render
# the Docs tab (markdown-it-py + mdit-py-plugins + pygments). Without
# this extra, build.py's ImportError fallback skips docs silently and
# `revalbench.com/docs/*` 404s.
set -euo pipefail

pip install uv
uv pip install --system -e ".[docs]"
python -m reval.cli leaderboard build
