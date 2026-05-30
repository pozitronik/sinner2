from installer import detect
from installer.detect import GpuInfo, SystemInfo


class TestParseGpus:
    def test_three_column(self):
        gpus = detect.parse_gpus("NVIDIA GeForce RTX 5090, 560.94, 12.0\n")
        assert gpus == [GpuInfo("NVIDIA GeForce RTX 5090", "560.94", "12.0")]

    def test_two_column_no_compute_cap(self):
        gpus = detect.parse_gpus("NVIDIA GeForce RTX 3080, 470.10\n")
        assert gpus == [GpuInfo("NVIDIA GeForce RTX 3080", "470.10", None)]

    def test_na_compute_cap_becomes_none(self):
        gpus = detect.parse_gpus("Tesla T4, 525.85, [N/A]\n")
        assert gpus[0].compute_capability is None

    def test_multiple_gpus(self):
        gpus = detect.parse_gpus("A, 560.0, 8.9\nB, 560.0, 8.9\n")
        assert [g.name for g in gpus] == ["A", "B"]

    def test_skips_blank_and_garbage_lines(self):
        gpus = detect.parse_gpus("\n  \nNVIDIA, 560.0, 8.9\n,,\n")
        assert len(gpus) == 1 and gpus[0].name == "NVIDIA"

    def test_empty_output(self):
        assert detect.parse_gpus("") == []


class TestPlatformDetection:
    def test_os(self, monkeypatch):
        monkeypatch.setattr(detect.sys, "platform", "win32")
        assert detect.detect_os() == "windows"
        monkeypatch.setattr(detect.sys, "platform", "darwin")
        assert detect.detect_os() == "macos"
        monkeypatch.setattr(detect.sys, "platform", "linux")
        assert detect.detect_os() == "linux"

    def test_arch_normalises(self, monkeypatch):
        monkeypatch.setattr(detect.platform, "machine", lambda: "AMD64")
        assert detect.detect_arch() == "x86_64"
        monkeypatch.setattr(detect.platform, "machine", lambda: "aarch64")
        assert detect.detect_arch() == "arm64"

    def test_wsl_false_on_non_linux(self, monkeypatch):
        monkeypatch.setattr(detect.sys, "platform", "win32")
        assert detect.detect_wsl() is False


class TestSystemInfo:
    def test_has_gpu_and_driver(self):
        info = SystemInfo(
            "linux", "x86_64", (GpuInfo("X", "560.0", "12.0"),), is_wsl=False
        )
        assert info.has_nvidia_gpu is True
        assert info.driver_version == "560.0"

    def test_no_gpu(self):
        info = SystemInfo("linux", "x86_64", (), is_wsl=False)
        assert info.has_nvidia_gpu is False
        assert info.driver_version is None
