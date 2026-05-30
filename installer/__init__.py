"""Standalone install + manage tool for sinner2.

stdlib-only by design: the launcher scripts run it under a bare uv-managed
Python (`uv run --no-project --python 3.12 installer/wizard.py`) BEFORE the app
or its dependencies exist, so it must not import anything outside the stdlib.
"""
