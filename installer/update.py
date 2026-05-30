"""GitHub-releases update check for the installed checkout.

The version logic (parse / compare / read the local version / turn a release
into an UpdateInfo) is pure and unit-tested; only the release fetch and the
`git pull` shell out. stdlib-only (urllib + tomllib), matching the rest of
installer/.

The update model is git-based: the installer files live inside a clone of
`pozitronik/sinner2`, so an update is `git pull --ff-only` followed by a
repair (re-sync of dependencies for the new code). When the project dir is not
a git checkout we can only point the user at the release download page.
"""
from __future__ import annotations

import json
import re
import subprocess
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = "pozitronik/sinner2"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"


@dataclass(frozen=True)
class UpdateInfo:
    current: str
    latest: str
    url: str
    notes: str


# ---- Pure version logic (tested) ----

def parse_version(text: str | None) -> tuple[int, ...]:
    """Lenient parse: 'v1.2.3' -> (1, 2, 3). A leading 'v' is dropped and any
    pre-release / build suffix ('-rc1', '+meta') is ignored. Unparseable or
    empty input -> () (which compares as the lowest possible version)."""
    if not text:
        return ()
    match = re.match(r"\d+(?:\.\d+)*", text.strip().lstrip("vV"))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def is_newer(latest: str | None, current: str | None) -> bool:
    """True only when `latest` parses to a strictly greater version. An
    unparseable remote tag never triggers an update prompt."""
    latest_v = parse_version(latest)
    if not latest_v:
        return False
    return latest_v > parse_version(current)


def installed_version(project_dir: Path) -> str | None:
    """The checked-out [project].version from pyproject.toml (source of truth),
    or None if it can't be read."""
    try:
        data = tomllib.loads((project_dir / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    version = data.get("project", {}).get("version")
    return version if isinstance(version, str) else None


def release_to_info(release: dict, current: str) -> UpdateInfo | None:
    """Turn a GitHub release object into UpdateInfo, but only if it is newer
    than `current`; otherwise None."""
    tag = (release or {}).get("tag_name") or ""
    if not is_newer(tag, current):
        return None
    return UpdateInfo(
        current=current,
        latest=tag,
        url=(release.get("html_url") or RELEASES_PAGE),
        notes=(release.get("body") or "").strip(),
    )


# ---- Network + git (best-effort, never raise) ----

def fetch_latest_release(url: str = LATEST_RELEASE_URL, timeout: float = 10.0) -> dict | None:
    """The 'latest release' JSON, or None on any failure — offline, rate
    limited, or no release published yet (the endpoint 404s, which urllib
    raises as an HTTPError, a URLError subclass)."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sinner2-installer",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def check_for_update(project_dir: Path, fetcher=fetch_latest_release) -> UpdateInfo | None:
    """Combine the local version and the latest release into an UpdateInfo, or
    None when up to date / offline / version unreadable. `fetcher` is injected
    so the logic is testable without the network."""
    current = installed_version(project_dir)
    if not current:
        return None
    release = fetcher()
    if not release:
        return None
    return release_to_info(release, current)


def is_git_checkout(project_dir: Path) -> bool:
    return (project_dir / ".git").exists()


def git_pull(project_dir: Path) -> tuple[bool, str]:
    """Fast-forward the checkout. Returns (ok, combined_output) and never
    raises — a dirty tree / diverged branch surfaces as ok=False so the caller
    can ask the user to resolve it by hand."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return False, f"git pull could not run: {exc}"
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
