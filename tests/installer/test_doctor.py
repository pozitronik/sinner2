from installer import doctor
from installer.doctor import CheckResult


class TestParseProbeOutput:
    def test_valid_json_last_line(self):
        out = 'some noise\n{"python": "3.12.3", "torch": "2.7.0"}\n'
        assert doctor.parse_probe_output(out) == {"python": "3.12.3", "torch": "2.7.0"}

    def test_no_json(self):
        assert doctor.parse_probe_output("just noise\n") == {}

    def test_invalid_json(self):
        assert doctor.parse_probe_output("{not json}") == {}


def _by_name(results, name):
    return next(r for r in results if r.name == name)


class TestInterpret:
    def _cuda_ok(self):
        return {
            "python": "3.12.3",
            "torch": "2.7.0",
            "torch_cuda": True,
            "device": "RTX 5090",
            "ort": "1.20.0",
            "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "sinner2": True,
        }

    def test_cuda_all_pass(self):
        results = doctor.interpret(self._cuda_ok(), "cuda")
        assert doctor.all_ok(results)
        assert _by_name(results, "CUDA via torch").ok
        assert _by_name(results, "CUDA execution provider").ok

    def test_cpu_variant_skips_cuda_checks(self):
        data = {
            "python": "3.12.3",
            "torch": "2.7.0",
            "ort": "1.20.0",
            "ort_providers": ["CPUExecutionProvider"],
            "sinner2": True,
        }
        results = doctor.interpret(data, "cpu")
        names = [r.name for r in results]
        assert "CUDA via torch" not in names
        assert "CUDA execution provider" not in names
        assert doctor.all_ok(results)

    def test_cuda_torch_unavailable_fails(self):
        data = self._cuda_ok()
        data["torch_cuda"] = False
        data["device"] = None
        results = doctor.interpret(data, "cuda")
        assert not doctor.all_ok(results)
        assert "driver" in _by_name(results, "CUDA via torch").detail

    def test_cuda_provider_missing_fails(self):
        data = self._cuda_ok()
        data["ort_providers"] = ["CPUExecutionProvider"]
        results = doctor.interpret(data, "cuda")
        check = _by_name(results, "CUDA execution provider")
        assert not check.ok
        assert "onnxruntime-gpu" in check.detail

    def test_torch_import_error_fails(self):
        data = {"python": "3.12.3", "torch_error": "No module named 'torch'", "sinner2": True}
        results = doctor.interpret(data, "cpu")
        assert not _by_name(results, "PyTorch").ok

    def test_sinner2_import_failure(self):
        data = {"python": "3.12.3", "torch": "2.7.0", "ort": "1.20.0", "sinner2_error": "boom"}
        results = doctor.interpret(data, "cpu")
        assert not _by_name(results, "sinner2 import").ok

    def test_wrong_python_version_fails(self):
        data = {"python": "3.10.0", "torch": "2.7.0", "ort": "1.20.0", "sinner2": True}
        results = doctor.interpret(data, "cpu")
        assert not _by_name(results, "Python 3.12").ok
