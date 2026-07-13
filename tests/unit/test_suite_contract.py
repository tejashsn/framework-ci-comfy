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
sys.path.insert(0, str(REPO / "scripts"))
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


def test_expand_selection_requires_explicit_test_names():
    import create_config
    import yaml
    cfg = yaml.safe_load((SUITE / "config" / "models_config.yaml").read_text())
    with pytest.raises(ValueError, match="test_names required"):
        create_config.expand_selection("", cfg)
    with pytest.raises(ValueError, match="Unknown test name"):
        create_config.expand_selection("comfyui_not_a_test", cfg)
    selected = create_config.expand_selection(
        "comfyui_stable_diffusion_2_1,comfyui_flux1_dev", cfg
    )
    assert selected == ["comfyui_stable_diffusion_2_1", "comfyui_flux1_dev"]


# --------------------------------------------------------------------------- #
# ported executors import under the new layout
# --------------------------------------------------------------------------- #
def test_ported_executors_import():
    import comfyui_runtime  # noqa: F401
    import model_check
    import fetch_models
    import comfyui_validator  # noqa: F401
    # model_check must resolve the relocated config/models.json
    assert len(model_check.load_models_manifest()) > 0
    assert fetch_models.auto_fetch_enabled() is True


def test_fetch_models_no_source(tmp_path, monkeypatch):
    import fetch_models
    monkeypatch.setenv("AUTO_FETCH_MODELS", "true")
    manifest = {"missing.safetensors": {"source": None, "subdirs": ["checkpoints"]}}
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    ok, detail = fetch_models.download_one(
        "missing.safetensors", ["checkpoints"], str(comfy), manifest,
    )
    assert not ok
    assert "no download source" in detail


def test_fetch_models_already_present(tmp_path):
    import fetch_models
    comfy = tmp_path / "ComfyUI"
    ckpt = comfy / "models" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "present.safetensors").write_bytes(b"x")
    manifest = {
        "present.safetensors": {
            "source": {"type": "huggingface", "repo_id": "x/y", "filename": "present.safetensors"},
            "subdirs": ["checkpoints"],
        }
    }
    ok, detail = fetch_models.download_one(
        "present.safetensors", ["checkpoints"], str(comfy), manifest,
    )
    assert ok and "already" in detail


def test_auto_fetch_disabled(monkeypatch):
    import fetch_models
    monkeypatch.setenv("AUTO_FETCH_MODELS", "false")
    assert fetch_models.auto_fetch_enabled() is False


def test_artifactory_model_url(monkeypatch):
    import fetch_models
    monkeypatch.setenv("ARTIFACTORY_URL", "https://art.example.com")
    monkeypatch.setenv("COMFYUI_MODELS_ARTIFACTORY_REPO", "artifactory/rocm-qa-model-cache/comfyui")
    entry = {
        "artifactory": {"path": "checkpoints/foo.safetensors"},
        "subdirs": ["checkpoints"],
    }
    url = fetch_models.artifactory_model_url(entry, "foo.safetensors", ["checkpoints"])
    assert url == (
        "https://art.example.com/artifactory/rocm-qa-model-cache/comfyui/"
        "checkpoints/foo.safetensors"
    )


def test_artifactory_requires_credentials(monkeypatch, tmp_path):
    import fetch_models
    monkeypatch.delenv("ARTIFACTORY_USER", raising=False)
    monkeypatch.delenv("ARTIFACTORY_PASSWORD", raising=False)
    comfy = tmp_path / "ComfyUI"
    manifest = {
        "gated.safetensors": {
            "source": {"type": "huggingface", "repo_id": "x/y", "filename": "gated.safetensors", "gated": True},
            "artifactory": {"path": "checkpoints/gated.safetensors"},
            "subdirs": ["checkpoints"],
        }
    }
    ok, detail = fetch_models.download_one(
        "gated.safetensors", ["checkpoints"], str(comfy), manifest,
    )
    assert not ok
    assert "ARTIFACTORY_USER" in detail


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


# --------------------------------------------------------------------------- #
# prefetch_models: bulk model prefetch before the benchmark matrix
# --------------------------------------------------------------------------- #
def test_prefetch_collect_missing_union(tmp_path):
    import prefetch_models
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "checkpoints").mkdir(parents=True)
    names = ["comfyui_stable_diffusion_2_1", "comfyui_flux1_schnell"]
    missing = prefetch_models.collect_missing_for_tests(names, str(comfy))
    files = {m[0] for m in missing}
    assert "v1-5-pruned-emaonly-fp16.safetensors" in files
    assert "flux1-schnell-fp8.safetensors" in files


def test_prefetch_skips_when_all_present(tmp_path, monkeypatch):
    import prefetch_models
    monkeypatch.setenv("AUTO_FETCH_MODELS", "true")
    comfy = tmp_path / "ComfyUI"
    ckpt = comfy / "models" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "v1-5-pruned-emaonly-fp16.safetensors").write_bytes(b"x")
    rc = prefetch_models.prefetch_tests(
        ["comfyui_stable_diffusion_2_1"], str(comfy),
    )
    assert rc == 0


def test_prefetch_disabled_is_noop(tmp_path, monkeypatch):
    import prefetch_models
    monkeypatch.setenv("AUTO_FETCH_MODELS", "false")
    rc = prefetch_models.prefetch_tests(
        ["comfyui_stable_diffusion_2_1"], str(tmp_path / "ComfyUI"),
    )
    assert rc == 0


# --------------------------------------------------------------------------- #
# workflow inputs + failure classification + test requirements
# --------------------------------------------------------------------------- #
def test_workflow_input_check_referenced_and_missing(tmp_path):
    import workflow_input_check
    comfy = tmp_path / "ComfyUI"
    (comfy / "input").mkdir(parents=True)
    wf = SUITE / "workflows" / "wan2.2_14b_i2v.json"
    refs = workflow_input_check.referenced_inputs(wf)
    assert "video_wan2_2_14B_i2v_input_image.jpg" in refs
    missing = workflow_input_check.missing_inputs(wf, str(comfy))
    assert "video_wan2_2_14B_i2v_input_image.jpg" in missing


def test_fetch_workflow_inputs_uses_bundled_assets(tmp_path, monkeypatch):
    import fetch_workflow_inputs
    monkeypatch.setenv("AUTO_FETCH_MODELS", "true")
    comfy = tmp_path / "ComfyUI"
    ok, detail = fetch_workflow_inputs.download_one(
        "video_wan2_2_14B_i2v_input_image.jpg", str(comfy), {},
    )
    assert ok, detail
    assert "bundled" in detail
    dest = comfy / "input" / "video_wan2_2_14B_i2v_input_image.jpg"
    bundled = fetch_workflow_inputs.bundled_asset_path(
        "video_wan2_2_14B_i2v_input_image.jpg")
    assert dest.is_file()
    assert dest.stat().st_size == bundled.stat().st_size


def test_failure_classify_server_crash():
    import failure_classify
    tail = "Requested to load ChromaRadiance\nPin error.\nFatal Python error: Aborted\n"
    category, reason, verdict = failure_classify.classify(
        "prompt abc did not complete within 1800s",
        server_tail=tail,
    )
    assert category == "server_crash"
    assert verdict == "INFRA_ERROR"
    assert "Pin error" in reason or "server_crash" in reason


def test_failure_classify_missing_input():
    import failure_classify
    category, _, verdict = failure_classify.classify(
        "submit/poll error: HTTP 400 Bad Request: Prompt outputs failed validation | "
        "LoadImage[img] Invalid image file: example.jpg",
    )
    assert category == "missing_input"
    assert verdict is None


def test_test_requirements_arch_and_vram_skip():
    import test_requirements
    test = {"os": ["linux"], "min_vram_mb": 999999, "exclude_archs": ["gfx9999"]}
    ok, reason = test_requirements.check_arch(test, "gfx9999")
    assert not ok and "exclude_archs" in reason
    ok, reason = test_requirements.check_requirements(
        test, gpu_arch="gfx1100", os_family="linux", python_exe=sys.executable)
    # VRAM may pass or be skipped on hosts without torch GPU; arch must pass.
    assert ok or "VRAM" in reason or "vram" in reason.lower()


def test_capture_runtime_environment_markdown_shape():
    import capture_runtime_environment as cre

    report = {
        "captured_at": "2026-07-13T00:00:00+00:00",
        "runner_name": "ctr-navi4x-at48-ws003",
        "metadata_file": "/c/TheRock/rocm-7.14.0rc2/metadata.conf",
        "rocm_stack": {
            "rocm_active": "7.14.0rc2 -> C:/TheRock/rocm-7.14.0rc2",
            "rocm_path": "C:/TheRock/rocm-7.14.0rc2",
            "hip_path_user": "C:/TheRock/rocm-7.14.0rc2",
            "hip_path_machine": "C:/TheRock/build",
            "source_tarball": "therock-dist-windows-gfx1153-7.14.0rc2.tar.gz",
            "therock_commit": "d1674be5a71ce9919474a4b2cbd335f387322822",
            "installed_version": "7.14.0rc2",
        },
        "python_ml_runtime": {
            "venv": "C:/TheRock/.venv314",
            "python_version": "3.14.6",
            "torch": "2.12.0+rocm7.14.0rc2",
            "torchvision": "0.27.0+rocm7.14.0rc2",
            "torchaudio": "2.11.0+rocm7.14.0rc2",
            "hip_torch": "7.14.60850",
            "rocm_sdk": "7.14.0rc2",
            "device_wheels": "amd-torch-device-gfx1153, rocm-sdk-device-gfx1153",
            "gpu_detected": True,
            "gpu_arch": "gfx1153",
        },
    }
    md = cre.format_markdown(report, expected_rocm="7.14.*")
    assert "ROCm (active)" in md
    assert "therock-dist-windows-gfx1153" in md
    assert "amd-torch-device-gfx1153" in md
    assert "MISMATCH" not in md


def test_capture_runtime_environment_rocm_drift():
    import capture_runtime_environment as cre

    assert cre._match_expected("7.14.0rc2", "7.14.*") is True
    assert cre._match_expected("7.15.0", "7.14.*") is False
    assert cre._match_expected("", "7.14.*") is None


def test_version_from_wheel_url():
    import capture_runtime_environment as cre

    url = (
        "https://artifactory.example.com/torch-2.12.0%2Brocm7.14.0rc2-"
        "cp314-cp314-win_amd64.whl"
    )
    assert cre._version_from_wheel_url(url, "torch") == "2.12.0+rocm7.14.0rc2"
    assert cre._wheel_versions_match("2.12.0+rocm7.14.0rc2", "2.12.0+rocm7.14.0rc2") is True
    assert cre._wheel_versions_match("2.11.0+rocm7.14.0rc2", "2.12.0+rocm7.14.0rc2") is False


def test_capture_runtime_environment_wheel_comparison_markdown():
    import capture_runtime_environment as cre

    report = cre.collect(sys.executable)
    wheels = cre.build_expected_wheels(
        torch_url="https://example.com/torch-2.12.0%2Brocm7.14.0rc2-cp314-cp314-win_amd64.whl",
        torchvision_url="https://example.com/torchvision-0.27.0%2Brocm7.14.0rc2-cp314-cp314-win_amd64.whl",
    )
    cre.attach_wheel_expectations(report, wheels)
    report["python_ml_runtime"]["torch"] = "2.12.0+rocm7.14.0rc2"
    report["python_ml_runtime"]["torchvision"] = "0.26.0+rocm7.14.0rc2"
    report["wheel_comparisons"]["torch"]["match"] = True
    report["wheel_comparisons"]["torchvision"]["match"] = False

    md = cre.format_markdown(report, expected_wheels=wheels)
    assert "Expected (workflow URL)" in md
    assert "example.com/torch-2.12.0" in md
    assert "**MISMATCH**" in md
    assert "### Wheel URLs (workflow inputs)" in md
