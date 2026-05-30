#!/usr/bin/env bash
# Re-runnable troubleshooter: re-verify the install (and the GPU path).
set -euo pipefail
cd "$(dirname "$0")"
exec uv run --no-project --python 3.12 installer/wizard.py --doctor "$@"
