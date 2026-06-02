from pathlib import Path

from installer import steps


class TestVariantMapping:
    def test_is_gpu_variant(self):
        assert steps.is_gpu_variant("cuda") is True
        assert steps.is_gpu_variant("cuda118") is True
        assert steps.is_gpu_variant("cpu") is False
        assert steps.is_gpu_variant("mac-arm") is False

    def test_torch_index_url(self):
        assert "cu128" in steps.torch_index_url("cuda")
        assert "cu118" in steps.torch_index_url("cuda118")
        assert "cpu" in steps.torch_index_url("cpu")
        assert steps.torch_index_url("mac-arm") is None


class TestPaths:
    def test_venv_python_windows(self):
        assert steps.venv_python(Path("/p/.venv"), "windows").name == "python.exe"
        assert "Scripts" in steps.venv_python(Path("/p/.venv"), "windows").parts

    def test_venv_python_unix(self):
        p = steps.venv_python(Path("/p/.venv"), "linux")
        assert p.parts[-2:] == ("bin", "python")

    def test_site_packages_windows(self):
        sp = steps.site_packages_dir(Path("/p/.venv"), "windows")
        assert sp.parts[-2:] == ("Lib", "site-packages")

    def test_site_packages_unix(self):
        sp = steps.site_packages_dir(Path("/p/.venv"), "linux")
        assert sp.parts[-3:] == ("lib", "python3.12", "site-packages")


class TestCommandBuilders:
    def test_create_venv(self):
        cmd = steps.create_venv_command("uv", Path(".venv"))
        assert cmd[:4] == ["uv", "venv", "--python", "3.12"]

    def test_torch_install_gpu_has_index(self):
        cmd = steps.torch_install_command("uv", Path("py"), "cuda")
        assert "torch" in cmd and "torchvision" in cmd
        assert "--index-url" in cmd
        assert cmd[cmd.index("--index-url") + 1].endswith("cu128")

    def test_torch_install_macarm_no_index(self):
        cmd = steps.torch_install_command("uv", Path("py"), "mac-arm")
        assert "--index-url" not in cmd

    def test_app_install_includes_extra_and_gui(self):
        cmd = steps.app_install_command("uv", Path("py"), "cuda")
        assert cmd[-1] == ".[cuda,gui]"
        assert "-e" in cmd

    def test_ort_gpu_reinstall(self):
        cmd = steps.ort_gpu_reinstall_command("uv", Path("py"))
        assert "--reinstall" in cmd and "--no-deps" in cmd
        assert cmd[-1] == "onnxruntime-gpu"  # default (cuda): latest CUDA-12 build

    def test_ort_gpu_reinstall_pins_for_cuda118(self):
        # The reinstall must honour the cuda118 extra's pin — an unpinned
        # reinstall grabs the latest (CUDA-12) build, overriding the CUDA-11.8
        # pin and silently falling back to CPU on older GPUs.
        cmd = steps.ort_gpu_reinstall_command("uv", Path("py"), "cuda118")
        assert cmd[-1] == "onnxruntime-gpu>=1.18,<1.19"

    def test_ort_gpu_reinstall_unpinned_for_cuda(self):
        cmd = steps.ort_gpu_reinstall_command("uv", Path("py"), "cuda")
        assert cmd[-1] == "onnxruntime-gpu"

    def test_tensorrt_install_pins_10x(self):
        cmd = steps.tensorrt_install_command("uv", Path("py"))
        # Must pin the 10.x major — onnxruntime's TRT EP links nvinfer_10.
        assert cmd[-1] == "tensorrt-cu12~=10.0"
        assert "pip" in cmd and "install" in cmd


class TestBasicsrPatch:
    _OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
    _NEW = "from torchvision.transforms.functional import rgb_to_grayscale"

    def test_applies_patch(self, tmp_path):
        f = tmp_path / "degradations.py"
        f.write_text(f"import x\n{self._OLD}\ny = 1\n", encoding="utf-8")
        assert steps.apply_basicsr_patch(f) is True
        text = f.read_text(encoding="utf-8")
        assert self._NEW in text
        assert "functional_tensor" not in text

    def test_idempotent(self, tmp_path):
        f = tmp_path / "degradations.py"
        f.write_text(f"{self._OLD}\n", encoding="utf-8")
        assert steps.apply_basicsr_patch(f) is True
        assert steps.apply_basicsr_patch(f) is False  # already patched

    def test_no_change_when_pattern_absent(self, tmp_path):
        f = tmp_path / "degradations.py"
        f.write_text("import something_else\n", encoding="utf-8")
        assert steps.apply_basicsr_patch(f) is False

    def test_find_degradations(self, tmp_path):
        target = tmp_path / "basicsr" / "data" / "degradations.py"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")
        assert steps.find_basicsr_degradations(tmp_path) == target

    def test_find_degradations_absent(self, tmp_path):
        assert steps.find_basicsr_degradations(tmp_path) is None
