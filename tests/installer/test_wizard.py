from pathlib import Path

from installer import plan, wizard
from installer.wizard import (
    Context,
    Wizard,
    applicable_variants,
    build_install_plan,
    existing_install,
    variant_menu,
    write_run_scripts,
)


def _ctx(os="linux", base=Path(".")):
    return Context(
        os=os, arch="x86_64", project_dir=base, venv_dir=base / ".venv", uv="uv"
    )


class TestApplicableVariants:
    def test_linux_windows(self):
        assert applicable_variants("linux") == ["cuda", "cuda118", "cpu"]
        assert applicable_variants("windows") == ["cuda", "cuda118", "cpu"]

    def test_macos(self):
        assert applicable_variants("macos") == ["mac-arm", "cpu"]


class TestVariantMenu:
    def test_marks_recommended(self):
        menu = variant_menu("linux", "cpu")
        labels = dict(menu)
        assert "(recommended)" in labels["cpu"]
        assert "(recommended)" not in labels["cuda"]

    def test_macos_options(self):
        assert [v for v, _ in variant_menu("macos", "mac-arm")] == ["mac-arm", "cpu"]


class TestExistingInstall:
    def test_detects_venv_python(self, tmp_path):
        venv = tmp_path / ".venv"
        py = wizard.steps.venv_python(venv, "linux")
        py.parent.mkdir(parents=True)
        py.write_text("")
        assert existing_install(venv, "linux") is True

    def test_absent(self, tmp_path):
        assert existing_install(tmp_path / "nope", "linux") is False


class TestBuildInstallPlan:
    def test_cuda_plan(self, tmp_path):
        plan_steps = build_install_plan("uv", tmp_path / ".venv", "cuda", "linux")
        torch_step = next(s for s in plan_steps if "PyTorch" in s.label)
        assert "cu128" in " ".join(torch_step.command)
        assert any("ONNX Runtime" in s.label for s in plan_steps)  # gpu reinstall
        assert any(s.command is None for s in plan_steps)  # basicsr patch step

    def test_cpu_plan_has_no_gpu_reinstall(self, tmp_path):
        plan_steps = build_install_plan("uv", tmp_path / ".venv", "cpu", "linux")
        assert not any("ONNX Runtime" in s.label for s in plan_steps)

    def test_macarm_torch_has_no_index(self, tmp_path):
        plan_steps = build_install_plan("uv", tmp_path / ".venv", "mac-arm", "macos")
        torch_step = next(s for s in plan_steps if "PyTorch" in s.label)
        assert "--index-url" not in torch_step.command


class TestRunScripts:
    def test_unix(self, tmp_path):
        scripts = write_run_scripts(tmp_path, "linux")
        assert scripts == [tmp_path / "run.sh"]
        content = (tmp_path / "run.sh").read_text()
        assert "sinner2.gui" in content and ".venv/bin/python" in content

    def test_windows(self, tmp_path):
        scripts = write_run_scripts(tmp_path, "windows")
        assert scripts == [tmp_path / "run.bat"]
        content = (tmp_path / "run.bat").read_text()
        assert "sinner2.gui" in content and "python.exe" in content


class TestSelectVariant:
    def test_empty_input_picks_recommended(self):
        w = Wizard(_ctx(os="linux"), ask=lambda _p: "", say=lambda _m: None)
        assert w._select_variant(plan.Recommendation("cuda", "x")) == "cuda"

    def test_numeric_choice(self):
        # linux menu order: cuda(1) cuda118(2) cpu(3)
        w = Wizard(_ctx(os="linux"), ask=lambda _p: "3", say=lambda _m: None)
        assert w._select_variant(plan.Recommendation("cuda", "x")) == "cpu"


class TestDriverGate:
    def test_cpu_variant_passes_through(self):
        from installer.detect import SystemInfo

        w = Wizard(_ctx(), ask=lambda _p: "", say=lambda _m: None)
        info = SystemInfo("linux", "x86_64", (), is_wsl=False)
        assert w._driver_gate("cpu", info) == "cpu"

    def test_no_gpu_then_choose_cpu(self):
        from installer.detect import SystemInfo

        w = Wizard(_ctx(), ask=lambda _p: "c", say=lambda _m: None)
        info = SystemInfo("linux", "x86_64", (), is_wsl=False)  # GPU variant but no GPU
        assert w._driver_gate("cuda", info) == "cpu"
