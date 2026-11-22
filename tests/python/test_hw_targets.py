"""Hardware registry (.github/hardware.json), SKIP_* variables and their consumers."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hw = _load_script("hw_targets", "hw_targets.py")
reach = _load_script("check_self_hosted_reachability", "check-self-hosted-reachability.py")
TARGETS = hw.load()
SKIP_VARS = {spec["skip_var"] for spec in TARGETS.values()}


def _yaml():
    return pytest.importorskip("yaml")


def _workflow(name: str) -> dict:
    return _yaml().safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_registry_is_valid():
    assert hw.validate(TARGETS) == []
    assert {"sm70", "sm86", "sm120", "gfx1012", "metal-arm", "metal-intel"} <= set(TARGETS)


def test_resolve_drops_skipped_targets_in_request_order():
    env = {"SKIP_SM120": "true", "SKIP_SM70": "false", "SKIP_SM86": ""}
    rows, skipped = hw.resolve(["sm120", "sm86", "sm70"], TARGETS, env)
    assert [r["target"] for r in rows] == ["sm86", "sm70"]
    assert skipped == [("sm120", "SKIP_SM120")]
    assert rows[0]["runs_on"] == TARGETS["sm86"]["runs_on"]


def test_skip_comparison_matches_github_expressions():
    # GitHub compares strings case-insensitively; anything but "true" keeps the target.
    for value, skipped in (("TRUE", True), ("True", True), ("1", False), ("yes", False), (" true", False)):
        assert hw.is_skipped(TARGETS["sm70"], {"SKIP_SM70": value}) is skipped


def test_unknown_target_is_an_error():
    with pytest.raises(SystemExit, match="unknown hardware target"):
        hw.resolve(["sm71"], TARGETS, {})


def test_matrix_output_all_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setenv("SKIP_SM70", "true")
    assert hw.main(["matrix", "sm70"]) == 0
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["any"] == "false"
    assert json.loads(outputs["matrix"]) == {"include": []}
    assert outputs["skipped"] == "sm70"


def test_reusable_workflow_maps_every_skip_var():
    env = _workflow("hw-targets.yml")["jobs"]["resolve"]["env"]
    assert set(env) == SKIP_VARS
    for var, value in env.items():
        assert value == f"${{{{ vars.{var} }}}}"


def test_actionlint_declares_every_skip_var():
    config = _yaml().safe_load((ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8"))
    assert SKIP_VARS <= set(config["config-variables"])


def test_workflows_only_reference_registered_skip_vars():
    used: set[str] = set()
    for path in WORKFLOWS.glob("*.yml"):
        used |= set(re.findall(r"vars\.(SKIP_[A-Z0-9_]+)", path.read_text(encoding="utf-8")))
    assert used <= SKIP_VARS
    assert used == SKIP_VARS, f"registered but never read: {sorted(SKIP_VARS - used)}"


def _from_json_runs_on(expr: str) -> dict:
    return json.loads(re.search(r"fromJSON\('(\{.*?\})'\)", expr).group(1))


@pytest.mark.parametrize(
    ("job", "target"),
    [
        ("bench-perf-v100s", "sm70"),
        ("bench-perf-rtx3070", "sm86"),
        ("bench-perf-isolated", "sm86"),
        ("bench-perf-rtx5060ti", "sm120"),
        ("bench-perf-gfx1012", "gfx1012"),
    ],
)
def test_bench_runs_on_matches_registry(job: str, target: str):
    jobs = _workflow("bench.yml")["jobs"]
    assert _from_json_runs_on(jobs[job]["runs-on"]) == TARGETS[target]["runs_on"]


@pytest.mark.parametrize(
    ("job", "target"),
    [("measure-v100s", "sm70"), ("measure-rtx3070", "sm86"), ("measure-rtx5060ti", "sm120")],
)
def test_bench_history_runs_on_matches_registry(job: str, target: str):
    assert _workflow("bench-history.yml")["jobs"][job]["runs-on"] == TARGETS[target]["runs_on"]


def test_smoke_amd_runs_on_matches_registry():
    assert _workflow("images.yml")["jobs"]["smoke-amd"]["runs-on"] == TARGETS["gfx1012"]["runs_on"]


@pytest.mark.parametrize(
    ("job", "var"),
    [
        ("bench-perf-v100s", "SKIP_SM70"),
        ("bench-perf-rtx3070", "SKIP_SM86"),
        ("bench-perf-rtx5060ti", "SKIP_SM120"),
        ("bench-perf-gfx1012", "SKIP_GFX1012"),
    ],
)
def test_bench_gpu_jobs_honour_skip(job: str, var: str):
    assert f"vars.{var} != 'true'" in _workflow("bench.yml")["jobs"][job]["if"]


def test_matrix_workflows_take_rows_from_the_registry():
    for name, job in (("gpu.yml", "gpu"), ("images.yml", "smoke"), ("gpu-diagnostics.yml", "diagnose")):
        spec = _workflow(name)["jobs"][job]
        assert spec["runs-on"] == "${{ matrix.runs_on }}", name
        assert "fromJSON(needs." in spec["strategy"]["matrix"], name
        assert "outputs.any == 'true'" in spec["if"], name


def test_nvidia_and_amd_targets_pin_a_group():
    # A labels-only runs-on lets any group holding the labels serve the job.
    for name in ("sm70", "sm86", "sm120", "gfx1012"):
        assert TARGETS[name]["runs_on"]["group"] == "public-ci", name


def test_dynamic_runs_on_counts_as_self_hosted(tmp_path: Path):
    yaml = _yaml()
    (tmp_path / "gpu.yml").write_text(
        yaml.safe_dump(
            {
                "on": ["pull_request"],
                "jobs": {
                    "gpu": {
                        "strategy": {"matrix": "${{ fromJSON(needs.targets.outputs.matrix) }}"},
                        "runs-on": "${{ matrix.runs_on }}",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    errors = reach.check_workflows(tmp_path)
    assert errors and "gpu" in errors[0]


def test_bench_comment_marks_disabled_hardware(tmp_path: Path):
    bench_comment = _load_script("bench_comment", "bench_comment.py")
    acc = tmp_path / "accuracy.json"
    acc.write_text(json.dumps({"kind": "accuracy", "verdict": "pass"}), encoding="utf-8")
    broken = tmp_path / "rtx5060ti.json"
    broken.write_text(json.dumps({"kind": "error", "verdict": "error", "error": "oom"}), encoding="utf-8")
    markdown, overall = bench_comment.render(
        jobs={"accuracy": acc, "rtx5060ti": broken, "gfx1012": None},
        run_url="",
        notices=[],
        disabled={"rtx5060ti": "SKIP_SM120", "gfx1012": "SKIP_GFX1012"},
    )
    assert overall == "pass"
    assert "hardware disabled (`SKIP_SM120=true`)" in markdown
    assert "hardware disabled (`SKIP_GFX1012=true`)" in markdown
    assert "oom" not in markdown
