"""Native-module variant selection (no Monte Carlo, CPU-only)."""

from __future__ import annotations

import sys

import pytest

from fuzzybits._runtime import _CUDA_SM_MOD, _select_backend, _select_variant


def test_auto_falls_back_to_generic_without_sm70(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: set())
    assert _select_variant("auto", {"generic": "g", "cuda-sm70": "c"}) == "generic"


def test_auto_selects_cuda_sm70_when_device_present(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: {70})
    assert _select_variant("auto", {"generic": "g", "cuda-sm70": "c"}) == "cuda-sm70"


def test_auto_matches_the_device_architecture(monkeypatch):
    import fuzzybits._runtime as rt

    installed = {"generic": "g", "cuda-sm70": "a", "cuda-sm86": "b", "cuda-sm120": "c"}
    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: {86})
    assert _select_variant("auto", installed) == "cuda-sm86"
    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: {120})
    assert _select_variant("auto", installed) == "cuda-sm120"
    # An sm_89 GPU has no matching AOT build; generic JITs for it.
    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: {89})
    assert _select_variant("auto", installed) == "generic"


def test_auto_uses_only_installed_variant(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "cuda_sm_versions", lambda: set())
    assert _select_variant("auto", {"cuda-sm70": "c"}) == "cuda-sm70"


def test_explicit_variant_requires_install():
    with pytest.raises(ImportError, match="cuda-sm70"):
        _select_variant("cuda-sm70", {"generic": "g"})
    with pytest.raises(ImportError, match="cuda-sm120"):
        _select_variant("cuda_sm120", {"generic": "g", "cuda-sm86": "b"})
    assert _select_variant("cuda_sm86", {"generic": "g", "cuda-sm86": "b"}) == "cuda-sm86"


def test_cuda_module_file_names():
    assert _CUDA_SM_MOD.match("_native_cuda_sm70.cpython-312-x86_64-linux-gnu.so").group(1) == "70"
    assert _CUDA_SM_MOD.match("_native_cuda_sm120.cp312-win_amd64.pyd").group(1) == "120"
    assert _CUDA_SM_MOD.match("_native_cuda_sm86.so").group(1) == "86"
    assert _CUDA_SM_MOD.match("_native_generic.cpython-312-x86_64-linux-gnu.so") is None


def test_backend_auto_cuda_if_nvidia(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_hip_device", lambda: False)
    monkeypatch.setattr(rt, "has_cuda_device", lambda: True)
    assert _select_backend("auto") == "cuda"
    monkeypatch.setattr(rt, "has_cuda_device", lambda: False)
    if sys.platform == "darwin":
        assert _select_backend("auto") == ""
    else:
        assert _select_backend("auto") == "omp"
    assert _select_backend("omp") == "omp"
    assert _select_backend("cuda") == "cuda"
    assert _select_backend("hip") == "hip"
    assert _select_backend("metal") == "metal"


def test_backend_auto_prefers_cuda_then_hip(monkeypatch):
    import fuzzybits._runtime as rt

    monkeypatch.setattr(rt, "has_cuda_device", lambda: False)
    monkeypatch.setattr(rt, "has_hip_device", lambda: True)
    assert _select_backend("auto") == "hip"
    monkeypatch.setattr(rt, "has_cuda_device", lambda: True)
    assert _select_backend("auto") == "cuda"


@pytest.mark.skipif(sys.platform != "linux", reason="KFD topology is Linux-only")
def test_amdgpu_gfx_targets_from_kfd_topology(monkeypatch, tmp_path):
    import fuzzybits._runtime as rt

    nodes = tmp_path / "nodes"
    for idx, version in enumerate((0, 100102, 90010, 110000)):
        node = nodes / str(idx)
        node.mkdir(parents=True)
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\nsimd_count 48\n")
    kfd = tmp_path / "kfd"
    kfd.touch()
    dri = tmp_path / "dri"
    dri.mkdir()
    monkeypatch.setattr(rt, "_KFD", kfd)
    monkeypatch.setattr(rt, "_KFD_TOPOLOGY", nodes)
    monkeypatch.setattr(rt, "_DRI", dri)
    # No render node: the KFD alone cannot submit work.
    assert rt.amdgpu_gfx_targets() == []
    (dri / "renderD128").touch()
    assert rt.amdgpu_gfx_targets() == ["gfx1012", "gfx90a", "gfx1100"]
    kfd.unlink()
    assert rt.amdgpu_gfx_targets() == []


def test_device_ordinal_follows_backend(monkeypatch):
    import fuzzybits._runtime as rt

    for name in ("ACPP_VISIBILITY_MASK", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    rt._apply_visibility("hip", 1)
    assert rt.os.environ["ACPP_VISIBILITY_MASK"] == "hip"
    assert rt.os.environ["HIP_VISIBLE_DEVICES"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in rt.os.environ
    rt._apply_visibility("cuda", 0)
    assert rt.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
