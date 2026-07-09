#!/usr/bin/env python3
"""
Unit tests for the framework-ci-comfy suite contract. No GPU / ComfyUI / network.
Covers:
  - models_config.yaml stays in sync with the authoritative manifest
  - comfyui_benchmark writes a RASTRA-shaped results_<test>.json
  - the results handler builds the fleet payload shape (deployment_info.execution_label)
  - ported executors import cleanly under the new layout
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "tests" / "src" / "inference" / "comfyui"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SUITE))
sys.path.insert(0, str(SUITE / "executors"))
sys.path.insert(0, str(SUITE / "scripts"))


# --------------------------------------------------------------------------- #
# config sync
# --------------------------------------------------------------------------- #
def test_models_config_in_sync_with_manifest():
    import create_config
    manifest = create_config.load_manifest()
    expected = create_config.build_models_config_text(manifest)
    actual = (SUITE / "config" / "models_config.yaml").read_text(encoding="utf-8")
    assert actual == expected, "models_config.yaml stale; run create_config.py --regenerate"


def test_models_config_covers_every_manifest_test():
    import yaml
    manifest = json.loads((SUITE / "config" / "suite_manifest.json").read_text())
    cfg = yaml.safe_load((SUITE / "config" / "models_config.yaml").read_text())
    man_names = {t["test_name"] for t in manifest["tests"]}
    cfg_names = {t["name"] for t in cfg["tests"]}
    assert cfg_names == man_names


# --------------------------------------------------------------------------- #
# ported executors import under the new layout
# --------------------------------------------------------------------------- #
def test_ported_executors_import():
    import comfyui_runtime  # noqa: F401
    import model_check
    import comfyui_validator  # noqa: F401
    # model_check must resolve the relocated config/models.json
    assert len(model_check.load_models_manifest()) > 0


# --------------------------------------------------------------------------- #
# benchmark selection + result shape
# --------------------------------------------------------------------------- #
def _load_cfg_and_manifest():
    import yaml
    cfg = yaml.safe_load((SUITE / "config" / "models_config.yaml").read_text())
    manifest = json.loads((SUITE / "config" / "suite_manifest.json").read_text())
    return cfg, manifest


def test_benchmark_selects_by_arch_and_tags():
    import comfyui_benchmark as cb
    cfg, manifest = _load_cfg_and_manifest()
    sel = cb.select_tests(cfg, manifest, model="", tags="smoke",
                          arch="gfx1100", os_family="linux")
    assert sel, "expected some smoke tests for gfx1100/linux"
    for e in sel:
        assert "smoke" in e.get("tags", [])
    # gfx908 should select the SD/LTX tests we added it to
    sel908 = cb.select_tests(cfg, manifest, model="", tags="",
                             arch="gfx908", os_family="linux")
    names = {e["name"] for e in sel908}
    assert "comfyui_stable_diffusion_2_1" in names


def test_write_result_json_shape(tmp_path):
    import comfyui_benchmark as cb
    # Fake an executor evidence dir with a summary.json + results.json.
    evi = tmp_path / "comfyui_stable_diffusion_2_1_20260101T000000Z"
    inner = evi / "comfyui_stable_diffusion_2_1_20260101T000000Z"
    inner.mkdir(parents=True)
    (inner / "summary.json").write_text(json.dumps({
        "test_name": "comfyui_stable_diffusion_2_1", "tms_key": "1002321",
        "verdict": "PASS", "duration_s": 60.0,
        "timestamp": "2026-01-01T00:00:00Z", "failure_reason": "",
    }))
    (inner / "results.json").write_text(json.dumps({"latency_avg_s": 16.2}))
    results_dir = tmp_path / "out"
    results_dir.mkdir()
    meta = {"arch": "gfx1100", "detected_arch": "gfx908", "rocm_version": "7.15.0",
            "device": "", "git_sha": "abc1234", "canonical": True,
            "workflow_hash": None}
    entry = {"name": "comfyui_stable_diffusion_2_1", "tags": ["smoke"]}
    out = cb.write_result_json(entry, "PASS", evi, results_dir, meta)
    doc = json.loads(Path(out).read_text())
    item = doc["results"][0]
    # RASTRA-shaped fields
    assert item["test_result"] == "PASS"
    assert item["result_status"] == "PASS"
    assert item["test_execution_time"] == 1.0   # 60s -> minutes
    assert item["test_config"]["detected_gpu_arch"] == "gfx908"
    assert item["test_config"]["requested_gpu_arch"] == "gfx1100"
    assert item["test_config"]["git_sha"] == "abc1234"
    prim = [m for m in item["test_metrics"] if m["primary"]]
    assert prim and prim[0]["score"] == 16.2 and prim[0]["unit"] == "seconds"


def test_skip_and_infra_map_to_result_status(tmp_path):
    import comfyui_benchmark as cb
    evi = tmp_path / "t_x"
    inner = evi / "t_x"
    inner.mkdir(parents=True)
    (inner / "summary.json").write_text(json.dumps({
        "test_name": "t", "verdict": "SKIP", "duration_s": 0,
        "failure_reason": "model missing"}))
    results_dir = tmp_path / "out"; results_dir.mkdir()
    out = cb.write_result_json({"name": "t", "tags": []}, "SKIP", evi,
                               results_dir, {"arch": "", "detected_arch": ""})
    item = json.loads(Path(out).read_text())["results"][0]
    # SKIP is not fake-greened: test_result FAIL-side but result_status preserved
    assert item["result_status"] == "SKIP"
    assert item["test_result"] == "FAIL"


# --------------------------------------------------------------------------- #
# results handler payload shape
# --------------------------------------------------------------------------- #
def test_handler_builds_fleet_payload(tmp_path):
    sys.path.insert(0, str(SUITE / "scripts"))
    import upload_from_artifacts
    from tests.utils.results import validate_payload

    rd = tmp_path / "logs" / "benchmark_results"
    rd.mkdir(parents=True)
    (tmp_path / "logs" / "bm_config.json").write_text(json.dumps({
        "os": "Ubuntu 24.04",
        "os_version": "24.04",
        "kernel": "6.8.0",
        "hostname": "test-host",
        "system_ip": "127.0.0.1",
        "cpu": {"model": "Test CPU", "cores": 64, "sockets": 1, "ram_size": 512},
        "gpu": {"count": 8, "name": "gfx942", "marketing_name": "MI300X"},
        "rocm_version": "7.15.0",
    }))
    (rd / "results_t.json").write_text(json.dumps({"results": [{
        "test_config": {"test_name": "comfyui_x", "sub_test_name": "comfyui_x"}, "test_metrics": [],
        "test_result": "PASS", "result_status": "PASS",
        "test_execution_time": 1.0, "test_start_time": "2026-01-01T00:00:00Z",
    }]}))
    payload, _ = upload_from_artifacts.build_payload_from_dir(
        rd, execution_label="sqa-nightly", logs_dir=tmp_path / "logs"
    )
    for k in ("test_environment", "bm_config", "deployment_info",
              "docker_config", "results", "test_app_commit", "test_app_version"):
        assert k in payload
    assert payload["deployment_info"]["execution_label"] == "sqa-nightly"
    assert validate_payload(payload)
    out = tmp_path / "payload.json"
    upload_from_artifacts.save_payload(payload, out)
    assert out.exists()


def test_framework_detection():
    from tests.utils.results.handler import _derive_framework
    assert _derive_framework("comfyui_stable_diffusion_2_1") == "comfyui"
    assert _derive_framework("vllm_serving_x") == "vllm"


# --------------------------------------------------------------------------- #
# artifactory upload path (no network)
# --------------------------------------------------------------------------- #
def test_prune_keeps_fastest_image_only(tmp_path):
    import single_test_protocol

    f1 = tmp_path / "a.png"
    f2 = tmp_path / "b.png"
    f3 = tmp_path / "c.png"
    for f in (f1, f2, f3):
        f.write_bytes(b"x")
    result = {
        "runs": [
            {"run": 1, "status": "OK", "latency_s": 10.0, "output_files": [str(f1)]},
            {"run": 2, "status": "OK", "latency_s": 5.0, "output_files": [str(f2)]},
            {"run": 3, "status": "OK", "latency_s": 8.0, "output_files": [str(f3)]},
        ],
        "errors": [],
    }
    single_test_protocol.prune_image_outputs_to_best_run(result, "image")
    assert result["output_files"] == [str(f2)]
    assert result["best_run"] == 2
    assert not f1.exists() and f2.exists() and not f3.exists()


def test_prune_skips_video(tmp_path):
    import single_test_protocol

    f1, f2 = tmp_path / "a.mp4", tmp_path / "b.mp4"
    f1.write_bytes(b"x")
    f2.write_bytes(b"y")
    result = {
        "runs": [
            {"run": 1, "status": "OK", "latency_s": 10.0, "output_files": [str(f1)]},
            {"run": 2, "status": "OK", "latency_s": 5.0, "output_files": [str(f2)]},
        ],
        "errors": [],
    }
    single_test_protocol.prune_image_outputs_to_best_run(result, "video")
    assert f1.exists() and f2.exists()
    assert "best_run" not in result
    sys.path.insert(0, str(REPO / "scripts"))
    import upload_to_artifactory

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "artifactory_path.txt").write_text(
        "/artifactory/rocm-qa-test-logs/test-logs/framework-ci/comfyui/7.15.0/host_0-comfyui_sd21/2026-07-08_12-00-00"
    )
    src = logs / "benchmark_results"
    src.mkdir()
    (src / "results_x.json").write_text("{}")

    base = upload_to_artifactory.resolve_upload_base_url(
        (logs / "artifactory_path.txt").read_text()
    )
    assert base.endswith("2026-07-08_12-00-00")

    upload_base, ok, fail = upload_to_artifactory.upload_directory(
        src, logs_dir=logs, dry_run=True
    )
    assert ok == 1 and fail == 0
    assert "comfyui_sd21" in upload_base
