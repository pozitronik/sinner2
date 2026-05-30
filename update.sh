#!/usr/bin/env bash
# Check GitHub releases and, if a newer version exists, pull + re-sync.
# Double-clickable on most desktops; or: bash update.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (one-time)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12 >/dev/null 2>&1 || true
exec uv run --no-project --python 3.12 installer/wizard.py --update "$@"
