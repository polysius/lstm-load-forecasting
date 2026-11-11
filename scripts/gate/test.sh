#!/usr/bin/env bash
# Run by the no-mistakes gate inside WSL2 (see .no-mistakes.yaml).
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/root/.venvs/no-mistakes-gate
uv run --no-sync pytest -m 'not slow and not integration' -q
