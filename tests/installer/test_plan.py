from installer.detect import GpuInfo, SystemInfo
from installer.plan import recommend


def _sys(os="linux", arch="x86_64", gpus=()):
    return SystemInfo(os=os, arch=arch, gpus=gpus, is_wsl=False)


def _gpu(name="GPU", driver="560.94", cc="8.9"):
    return GpuInfo(name=name, driver_version=driver, compute_capability=cc)


class TestRecommend:
    def test_no_gpu_is_cpu(self):
        r = recommend(_sys())
        assert r.variant == "cpu"
        assert r.gpu_blocked is False

    def test_apple_silicon_is_mac_arm(self):
        assert recommend(_sys(os="macos", arch="arm64")).variant == "mac-arm"

    def test_intel_mac_is_cpu(self):
        assert recommend(_sys(os="macos", arch="x86_64")).variant == "cpu"

    def test_modern_driver_is_cuda128(self):
        r = recommend(_sys(gpus=(_gpu(driver="560.94", cc="12.0"),)))
        assert r.variant == "cuda"
        assert "CUDA 12.8" in r.reason

    def test_mid_driver_is_cuda118(self):
        r = recommend(_sys(gpus=(_gpu(driver="470.10", cc="8.6"),)))
        assert r.variant == "cuda118"

    def test_old_driver_falls_back_to_cpu(self):
        r = recommend(_sys(gpus=(_gpu(driver="440.0", cc="7.5"),)))
        assert r.variant == "cpu"
        assert r.gpu_blocked is True

    def test_blackwell_with_old_driver_is_cpu_blocked(self):
        # sm_120 (RTX 50xx) has no CUDA 11.8 kernels, so an old driver means
        # the GPU can't be used at all — not a cu118 fallback.
        r = recommend(
            _sys(gpus=(_gpu(name="RTX 5090", driver="470.0", cc="12.0"),))
        )
        assert r.variant == "cpu"
        assert r.gpu_blocked is True
        assert "12.8" in r.reason

    def test_unreadable_driver_is_cpu_blocked(self):
        r = recommend(_sys(gpus=(_gpu(driver="not-a-version", cc="8.9"),)))
        assert r.variant == "cpu"
        assert r.gpu_blocked is True
