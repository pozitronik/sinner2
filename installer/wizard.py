"""Interactive install + manage wizard.

Pure, testable helpers (the install-plan builder, variant menu, run-script
writer, existing-install check) sit at the top; the interactive Wizard
orchestrates them with injectable I/O. Run via the launcher scripts:
`uv run --no-project --python 3.12 installer/wizard.py`. stdlib-only.
"""
from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from installer import detect, doctor, plan, steps, update

_VARIANT_LABELS = {
    "cuda": "CUDA 12.8 — NVIDIA GPU, fastest",
    "cuda118": "CUDA 11.8 — older NVIDIA GPU / driver",
    "cpu": "CPU only — no GPU needed, slower",
    "mac-arm": "Apple Silicon — Metal / CPU",
}

_DRIVER_HELP = (
    "Get the latest NVIDIA driver: https://www.nvidia.com/Download/index.aspx\n"
    "  (CUDA 12.8 needs driver branch ~525+; RTX 50xx / Blackwell needs ~570+.)"
)


@dataclass(frozen=True)
class Step:
    label: str
    command: list[str] | None  # None => the basicsr patch step


@dataclass
class Context:
    os: str
    arch: str
    project_dir: Path
    venv_dir: Path
    uv: str | None  # uv executable, or None if not found


# ---- Pure helpers (tested) ----

def applicable_variants(os: str) -> list[str]:
    if os == "macos":
        return ["mac-arm", "cpu"]
    return ["cuda", "cuda118", "cpu"]


def variant_menu(os: str, recommended: str) -> list[tuple[str, str]]:
    """[(variant, label)] for the chosen OS, the recommended one marked."""
    out = []
    for variant in applicable_variants(os):
        mark = "  (recommended)" if variant == recommended else ""
        out.append((variant, _VARIANT_LABELS[variant] + mark))
    return out


def existing_install(venv_dir: Path, os: str) -> bool:
    return steps.venv_python(venv_dir, os).is_file()


def build_install_plan(
    uv: str, venv_dir: Path, variant: str, os: str, project_dir: str = "."
) -> list[Step]:
    py = steps.venv_python(venv_dir, os)
    plan_steps = [
        Step("Create Python 3.12 environment", steps.create_venv_command(uv, venv_dir)),
        Step(f"Install PyTorch ({variant})", steps.torch_install_command(uv, py, variant)),
        Step("Install sinner2 + dependencies", steps.app_install_command(uv, py, variant, project_dir)),
        Step("Patch basicsr for modern torchvision", None),
    ]
    if steps.is_gpu_variant(variant):
        plan_steps.append(
            Step("Ensure GPU ONNX Runtime wins", steps.ort_gpu_reinstall_command(uv, py))
        )
    return plan_steps


def write_run_scripts(project_dir: Path, os: str) -> list[Path]:
    """Write double-clickable launchers that run the app directly from the
    installed venv (no uv / no sync needed at runtime)."""
    written = []
    if os == "windows":
        bat = project_dir / "run.bat"
        bat.write_text(
            '@echo off\r\n'
            'cd /d "%~dp0"\r\n'
            r'"%~dp0.venv\Scripts\python.exe" -m sinner2.gui %*' + "\r\n",
            encoding="utf-8",
        )
        written.append(bat)
    else:
        sh = project_dir / "run.sh"
        sh.write_text(
            '#!/usr/bin/env bash\n'
            'cd "$(dirname "$0")"\n'
            'exec ./.venv/bin/python -m sinner2.gui "$@"\n',
            encoding="utf-8",
        )
        sh.chmod(0o755)
        written.append(sh)
    return written


# ---- Interactive wizard ----

class Wizard:
    def __init__(
        self,
        ctx: Context,
        ask: Callable[[str], str] = input,
        say: Callable[[str], None] = print,
    ) -> None:
        self.ctx = ctx
        self.ask = ask
        self.say = say

    def run(self, doctor_only: bool = False, update_only: bool = False) -> int:
        if update_only:
            return self._update()
        if self.ctx.uv is None:
            self.say("uv was not found on PATH. Re-run the install launcher.")
            return 1
        if doctor_only:
            return 0 if self._doctor(self._guess_variant()) else 1
        if existing_install(self.ctx.venv_dir, self.ctx.os):
            return self._manage()
        return self._install()

    # -- install --

    def _install(self, variant: str | None = None) -> int:
        info = detect.detect()
        if variant is None:
            rec = plan.recommend(info)
            variant = self._select_variant(rec)
            variant = self._driver_gate(variant, info)
            if variant is None:
                self.say("Aborted.")
                return 1
        if not self._execute_plan(variant):
            return 1
        ok = self._doctor(variant)
        scripts = write_run_scripts(self.ctx.project_dir, self.ctx.os)
        self.say(f"\nWrote launcher: {', '.join(p.name for p in scripts)}")
        self.say(
            "Done. Models download on first launch."
            + ("" if ok else "  (Some checks failed — see above.)")
        )
        return 0 if ok else 1

    def _select_variant(self, rec: plan.Recommendation) -> str:
        menu = variant_menu(self.ctx.os, rec.variant)
        self.say(f"\nDetected: {rec.reason}")
        self.say("Choose the build to install:")
        for i, (_v, label) in enumerate(menu, 1):
            self.say(f"  {i}) {label}")
        default = next(
            (i for i, (v, _) in enumerate(menu, 1) if v == rec.variant), 1
        )
        while True:
            raw = self.ask(f"Choice [{default}]: ").strip()
            if not raw:
                return menu[default - 1][0]
            if raw.isdigit() and 1 <= int(raw) <= len(menu):
                return menu[int(raw) - 1][0]
            self.say("  Enter one of the listed numbers.")

    def _driver_gate(self, variant: str, info: detect.SystemInfo) -> str | None:
        if not steps.is_gpu_variant(variant):
            return variant
        if info.has_nvidia_gpu and not plan.recommend(info).gpu_blocked:
            return variant
        self.say("\n⚠ GPU build selected, but the NVIDIA driver looks missing or too old.")
        self.say("  " + _DRIVER_HELP)
        while True:
            choice = self.ask(
                "  [R]echeck after installing, use [C]PU instead, or [A]bort? "
            ).strip().lower()
            if choice.startswith("r"):
                info = detect.detect()
                if info.has_nvidia_gpu and not plan.recommend(info).gpu_blocked:
                    return variant
                self.say("  Still no usable GPU driver.")
            elif choice.startswith("c"):
                return "cpu"
            elif choice.startswith("a"):
                return None

    def _execute_plan(self, variant: str) -> bool:
        for step in build_install_plan(
            self.ctx.uv, self.ctx.venv_dir, variant, self.ctx.os
        ):
            self.say(f"\n→ {step.label}")
            if step.command is None:
                self._run_basicsr_patch()
                continue
            if steps.run(step.command) != 0:
                self.say(f"  Step failed: {step.label}")
                return False
        return True

    def _run_basicsr_patch(self) -> None:
        site = steps.site_packages_dir(self.ctx.venv_dir, self.ctx.os)
        target = steps.find_basicsr_degradations(site)
        if target is None:
            self.say("  basicsr not found — skipping patch.")
            return
        self.say("  patched." if steps.apply_basicsr_patch(target) else "  already patched.")

    def _doctor(self, variant: str) -> bool:
        py = steps.venv_python(self.ctx.venv_dir, self.ctx.os)
        self.say("\nVerifying install:")
        results = doctor.run_doctor(py, variant)
        for r in results:
            self.say(f"  [{'OK ' if r.ok else 'FAIL'}] {r.name}: {r.detail}")
        return doctor.all_ok(results)

    # -- manage (re-run menu) --

    _MANAGE = [
        ("Check for updates", "update"),
        ("Repair / re-sync", "repair"),
        ("Switch hardware variant", "switch"),
        ("Run doctor", "doctor"),
        ("Reinstall from scratch", "reinstall"),
        ("Uninstall", "uninstall"),
        ("Quit", "quit"),
    ]

    def _manage(self) -> int:
        self.say("\nsinner2 is already installed.")
        self._maybe_announce_update()
        self.say("What would you like to do?")
        for i, (label, _key) in enumerate(self._MANAGE, 1):
            self.say(f"  {i}) {label}")
        raw = self.ask("Choice: ").strip()
        if not (raw.isdigit() and 1 <= int(raw) <= len(self._MANAGE)):
            return 0
        action = self._MANAGE[int(raw) - 1][1]
        return self._dispatch(action)

    def _dispatch(self, action: str) -> int:
        variant = self._guess_variant()
        if action == "quit":
            return 0
        if action == "update":
            return self._update()
        if action == "doctor":
            return 0 if self._doctor(variant) else 1
        if action == "repair":
            return self._install(variant=variant)
        if action == "switch":
            return self._install()  # re-pick variant
        if action == "reinstall":
            shutil.rmtree(self.ctx.venv_dir, ignore_errors=True)
            return self._install()
        if action == "uninstall":
            shutil.rmtree(self.ctx.venv_dir, ignore_errors=True)
            self.say("Removed the environment.")
            return 0
        return 0

    def _guess_variant(self) -> str:
        # Best-effort: re-derive the recommendation for the doctor/repair path.
        return plan.recommend(detect.detect()).variant

    # -- updates --

    def _maybe_announce_update(self) -> None:
        """Best-effort one-liner on the manage screen; silent if offline, up to
        date, or anything goes wrong — it must never block the menu."""
        try:
            info = update.check_for_update(
                self.ctx.project_dir,
                fetcher=lambda: update.fetch_latest_release(timeout=4.0),
            )
        except Exception:
            return
        if info:
            self.say(
                f"  ✨ Update available: {info.current} → {info.latest} "
                "(choose 'Check for updates')"
            )

    def _update(self) -> int:
        self.say("\nChecking for updates…")
        info = update.check_for_update(self.ctx.project_dir)
        if info is None:
            self.say("  You're on the latest version (or no release is published yet).")
            return 0
        self.say(f"  Update available: {info.current} → {info.latest}")
        if info.notes:
            for line in info.notes.splitlines()[:8]:
                self.say(f"    {line}")
        self.say(f"  Release notes: {info.url}")
        if not update.is_git_checkout(self.ctx.project_dir):
            self.say("  This isn't a git checkout — download the new version from the page above.")
            return 0
        if not self._confirm("\n  Update now (git pull + repair)?"):
            self.say("  Skipped.")
            return 0
        ok, output = update.git_pull(self.ctx.project_dir)
        if output:
            for line in output.splitlines():
                self.say(f"  {line}")
        if not ok:
            self.say("  Update failed — resolve the git state above, then retry.")
            return 1
        if self.ctx.uv is None:
            self.say("  Code updated. Re-run the installer to re-sync dependencies.")
            return 0
        self.say("\nRe-syncing dependencies for the new version:")
        return self._install(variant=self._guess_variant())

    def _confirm(self, prompt: str) -> bool:
        return self.ask(f"{prompt} [Y/n] ").strip().lower() in ("", "y", "yes")


def _build_context() -> Context:
    project_dir = Path(__file__).resolve().parent.parent
    return Context(
        os=detect.detect_os(),
        arch=detect.detect_arch(),
        project_dir=project_dir,
        venv_dir=project_dir / ".venv",
        uv=shutil.which("uv"),
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return Wizard(_build_context()).run(
        doctor_only="--doctor" in argv,
        update_only="--update" in argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
