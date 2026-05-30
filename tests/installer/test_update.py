import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from installer import update


class TestParseVersion:
    def test_plain(self):
        assert update.parse_version("1.2.3") == (1, 2, 3)

    def test_v_prefix(self):
        assert update.parse_version("v0.1.0") == (0, 1, 0)

    def test_prerelease_suffix_ignored(self):
        assert update.parse_version("v2.0.0-rc1") == (2, 0, 0)

    def test_empty_and_garbage(self):
        assert update.parse_version("") == ()
        assert update.parse_version(None) == ()
        assert update.parse_version("nightly") == ()


class TestIsNewer:
    def test_strictly_greater(self):
        assert update.is_newer("0.2.0", "0.1.0") is True
        assert update.is_newer("v1.0.0", "0.9.9") is True

    def test_equal_or_older(self):
        assert update.is_newer("0.1.0", "0.1.0") is False
        assert update.is_newer("0.1.0", "0.2.0") is False

    def test_v_prefix_is_ignored_for_equality(self):
        assert update.is_newer("v1.0", "1.0") is False

    def test_unparseable_remote_never_updates(self):
        assert update.is_newer("nightly", "0.1.0") is False


class TestInstalledVersion:
    def test_returns_nearest_tag(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Proc(0, "v0.2.0\n"))
        assert update.installed_version(Path(".")) == "v0.2.0"

    def test_no_tags_returns_none(self, monkeypatch):
        # `git describe` exits non-zero with "No names found" when there are no tags
        monkeypatch.setattr(
            subprocess, "run", lambda *_a, **_k: _Proc(128, "", "fatal: No names found")
        )
        assert update.installed_version(Path(".")) is None

    def test_not_a_git_checkout_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *_a, **_k: _Proc(128, "", "fatal: not a git repository")
        )
        assert update.installed_version(Path(".")) is None

    def test_git_missing_returns_none(self, monkeypatch):
        def boom(*_a, **_k):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", boom)
        assert update.installed_version(Path(".")) is None

    def test_empty_stdout_returns_none(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Proc(0, "   \n"))
        assert update.installed_version(Path(".")) is None


class TestReleaseToInfo:
    def test_newer_builds_info(self):
        info = update.release_to_info(
            {"tag_name": "v0.2.0", "html_url": "http://x", "body": " notes "}, "0.1.0"
        )
        assert info == update.UpdateInfo("0.1.0", "v0.2.0", "http://x", "notes")

    def test_not_newer_is_none(self):
        assert update.release_to_info({"tag_name": "v0.1.0"}, "0.1.0") is None

    def test_missing_url_falls_back_to_releases_page(self):
        info = update.release_to_info({"tag_name": "v9.0.0"}, "0.1.0")
        assert info is not None
        assert info.url == update.RELEASES_PAGE
        assert info.notes == ""


class TestCheckForUpdate:
    def _git_tag(self, monkeypatch, tag):
        # installed_version() shells out to `git describe --tags --abbrev=0`
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Proc(0, tag + "\n"))

    def test_newer_release(self, monkeypatch):
        self._git_tag(monkeypatch, "v0.1.0")
        info = update.check_for_update(
            Path("."), fetcher=lambda: {"tag_name": "v0.2.0", "html_url": "u", "body": "n"}
        )
        assert info is not None
        # current is now the raw tag (with the leading 'v'); comparison strips it
        assert (info.current, info.latest) == ("v0.1.0", "v0.2.0")

    def test_up_to_date(self, monkeypatch):
        self._git_tag(monkeypatch, "v0.2.0")
        assert update.check_for_update(Path("."), fetcher=lambda: {"tag_name": "v0.2.0"}) is None

    def test_offline(self, monkeypatch):
        self._git_tag(monkeypatch, "v0.1.0")
        assert update.check_for_update(Path("."), fetcher=lambda: None) is None

    def test_unknown_local_version(self, monkeypatch):
        # no tags yet -> installed_version() is None -> never prompts
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Proc(128, "", "No names found"))
        assert update.check_for_update(Path("."), fetcher=lambda: {"tag_name": "v9"}) is None


class _Resp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class TestFetchLatestRelease:
    def test_success(self, monkeypatch):
        payload = json.dumps({"tag_name": "v1.0.0"}).encode("utf-8")
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Resp(payload))
        assert update.fetch_latest_release() == {"tag_name": "v1.0.0"}

    def test_http_404_no_release(self, monkeypatch):
        def raise_http(*_a, **_k):
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

        monkeypatch.setattr(urllib.request, "urlopen", raise_http)
        assert update.fetch_latest_release() is None

    def test_offline(self, monkeypatch):
        def raise_url(*_a, **_k):
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(urllib.request, "urlopen", raise_url)
        assert update.fetch_latest_release() is None

    def test_invalid_json(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Resp(b"not json"))
        assert update.fetch_latest_release() is None


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestGitPull:
    def test_ok(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *_a, **_k: _Proc(0, "Already up to date.\n")
        )
        ok, out = update.git_pull(Path("."))
        assert ok is True
        assert "up to date" in out.lower()

    def test_failure_surfaces_output(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", lambda *_a, **_k: _Proc(1, "", "would be overwritten")
        )
        ok, out = update.git_pull(Path("."))
        assert ok is False
        assert "overwritten" in out

    def test_git_missing(self, monkeypatch):
        def boom(*_a, **_k):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", boom)
        ok, out = update.git_pull(Path("."))
        assert ok is False
        assert "could not run" in out


class TestIsGitCheckout:
    def test_true(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert update.is_git_checkout(tmp_path) is True

    def test_false(self, tmp_path):
        assert update.is_git_checkout(tmp_path) is False
