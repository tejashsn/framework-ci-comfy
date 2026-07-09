#!/usr/bin/env python3
"""
Layer    : Pre-run content gate (offline)
Purpose  : Decide whether a test should be SKIPped because its model weights are
           not present, vs. run for real. ComfyUI workflows reference model files
           by name in their loader nodes (e.g. UNETLoader.unet_name = "foo.safetensors").
           Those weights are large and often license-gated; when AUTO_FETCH_MODELS
           is enabled (default) missing files are downloaded from config/models.json.

Maps the common ComfyUI loader node types to the ComfyUI/models/<subdir>/ they
load from, resolves each referenced filename, and reports which are missing.

Usable as a library (missing_models / models_root_from_url) or as a CLI:
  python scripts/model_check.py --workflow workflows/chroma.json --comfyui_path ~/ComfyUI
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# framework-ci-comfy: models.json lives in the suite's config/ dir.
MODELS_JSON = REPO_ROOT / "config" / "models.json"

# class_type -> {input_field: [candidate model subdirs]}. Multiple subdirs are
# tried in order because ComfyUI has renamed some folders across versions
# (e.g. unet -> diffusion_models, clip -> text_encoders).
LOADER_MAP = {
    "CheckpointLoaderSimple":   {"ckpt_name": ["checkpoints"]},
    "CheckpointLoader":         {"ckpt_name": ["checkpoints"]},
    "unCLIPCheckpointLoader":   {"ckpt_name": ["checkpoints"]},
    "ImageOnlyCheckpointLoader": {"ckpt_name": ["checkpoints"]},
    "UNETLoader":               {"unet_name": ["unet", "diffusion_models"]},
    "VAELoader":                {"vae_name": ["vae"]},
    "CLIPLoader":               {"clip_name": ["clip", "text_encoders"]},
    "DualCLIPLoader":           {"clip_name1": ["clip", "text_encoders"],
                                 "clip_name2": ["clip", "text_encoders"]},
    "TripleCLIPLoader":         {"clip_name1": ["clip", "text_encoders"],
                                 "clip_name2": ["clip", "text_encoders"],
                                 "clip_name3": ["clip", "text_encoders"]},
    "CLIPVisionLoader":         {"clip_name": ["clip_vision"]},
    "LoraLoader":               {"lora_name": ["loras"]},
    "LoraLoaderModelOnly":      {"lora_name": ["loras"]},
    "ControlNetLoader":         {"control_net_name": ["controlnet"]},
    "DiffControlNetLoader":     {"control_net_name": ["controlnet"]},
    "UpscaleModelLoader":       {"model_name": ["upscale_models"]},
    "StyleModelLoader":         {"style_model_name": ["style_models"]},
    "GLIGENLoader":             {"gligen_name": ["gligen"]},
    "VAEApprox":                {"vae_name": ["vae_approx"]},
}


def _load_graph(workflow_path):
    data = json.loads(Path(workflow_path).read_text())
    if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data if isinstance(data, dict) else {}


def referenced_models(workflow_path):
    """Return a list of (filename, [candidate_subdirs]) referenced by loader
    nodes in the workflow. Filenames are exactly as ComfyUI expects them
    (may include a subfolder prefix like 'sdxl/foo.safetensors')."""
    graph = _load_graph(workflow_path)
    refs = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        fields = LOADER_MAP.get(node.get("class_type"))
        if not fields:
            continue
        inputs = node.get("inputs", {})
        for field, subdirs in fields.items():
            val = inputs.get(field)
            # Only consider literal filenames (a list value is a node link, not a file).
            if isinstance(val, str) and val.strip():
                refs.append((val.strip(), subdirs))
    return refs


def _exists_in(models_root, filename, subdirs):
    """True if `filename` exists under any of the candidate subdirs (filenames
    may themselves contain a subfolder, which ComfyUI joins onto the subdir)."""
    for sub in subdirs:
        if (models_root / sub / filename).is_file():
            return True
    return False


def missing_models(workflow_path, comfy_path):
    """Return a list of (filename, subdirs) for models that are referenced by
    the workflow but not found under ComfyUI/models/<subdir>/.

    If comfy_path is unknown or has no models/ dir, returns [] (we can't tell, so
    we let the run proceed and let ComfyUI report the real error)."""
    if not comfy_path:
        return []
    models_root = Path(comfy_path) / "models"
    if not models_root.exists():
        return []
    missing = []
    for filename, subdirs in referenced_models(workflow_path):
        if not _exists_in(models_root, filename, subdirs):
            missing.append((filename, subdirs))
    return missing


def skip_message(missing):
    """Human-readable SKIP message for a list of missing (filename, subdirs)."""
    parts = []
    for filename, subdirs in missing:
        loc = f"models/{subdirs[0]}/{filename}"
        parts.append(loc)
    joined = ", ".join(parts)
    return (f"SKIP: model not found: {joined}. "
            "Place the file(s) under ComfyUI/models/, set HF_TOKEN for gated "
            "models, or enable AUTO_FETCH_MODELS (default) to download.")


def load_models_manifest(path=MODELS_JSON):
    """Load configs/models.json ({filename: {sha256, size_bytes, source}}).
    Returns {} if absent/unreadable - identity checks then degrade to presence."""
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_path(models_root, filename, subdirs):
    """Return the on-disk Path of a present model file, or None if not found."""
    for sub in subdirs:
        cand = models_root / sub / filename
        if cand.is_file():
            return cand
    return None


def _sha256_of(path, chunk=1 << 20):
    """Streaming sha256 of a file, or None on read error."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def verify_identity(filename, path, manifest):
    """Verify a PRESENT model file against configs/models.json.
    Returns (ok, reason):
      * filename absent from manifest, or size/sha256 both null -> (True, "")
        (presence-only; we have nothing to verify against).
      * size mismatch (checked first, cheap)   -> (False, "IDENTITY_MISMATCH ...")
      * sha256 mismatch (only if size ok/null)  -> (False, "IDENTITY_MISMATCH ...")
      * all provided fields match               -> (True, "")
    Never a FAIL - a mismatch is a content SKIP (wrong file, re-provision it).
    """
    entry = manifest.get(filename)
    if not entry:
        return True, ""
    exp_size = entry.get("size_bytes")
    exp_sha = entry.get("sha256")
    if exp_size is None and exp_sha is None:
        return True, ""  # nothing to verify yet (checksums filled on the fleet)

    if exp_size is not None:
        try:
            actual_size = Path(path).stat().st_size
        except OSError as e:
            return False, f"IDENTITY_MISMATCH {filename}: cannot stat ({e})"
        if actual_size != exp_size:
            return False, (f"IDENTITY_MISMATCH {filename}: size {actual_size} "
                           f"!= expected {exp_size}")

    if exp_sha is not None:
        actual_sha = _sha256_of(path)
        if actual_sha is None:
            return False, f"IDENTITY_MISMATCH {filename}: cannot read for sha256"
        if actual_sha.lower() != str(exp_sha).lower():
            return False, (f"IDENTITY_MISMATCH {filename}: sha256 "
                           f"{actual_sha[:12]}... != expected "
                           f"{str(exp_sha)[:12]}...")
    return True, ""


def identity_mismatches(workflow_path, comfy_path, manifest=None):
    """For every referenced model that IS present, verify it against models.json.
    Returns a list of mismatch reason strings (empty when all present files pass
    or have no checksums to check). Missing files are handled by missing_models,
    not here."""
    if not comfy_path:
        return []
    models_root = Path(comfy_path) / "models"
    if not models_root.exists():
        return []
    if manifest is None:
        manifest = load_models_manifest()
    reasons = []
    for filename, subdirs in referenced_models(workflow_path):
        path = _resolve_path(models_root, filename, subdirs)
        if path is None:
            continue  # absent -> presence check's job, not identity's
        ok, reason = verify_identity(filename, path, manifest)
        if not ok:
            reasons.append(reason)
    return reasons


def main():
    p = argparse.ArgumentParser(description="ComfyUI workflow model presence check")
    p.add_argument("--workflow", required=True)
    p.add_argument("--comfyui_path", default=None,
                   help="ComfyUI install dir (the one containing models/).")
    p.add_argument("--verify", action="store_true",
                   help="Also verify PRESENT files against configs/models.json "
                        "(size then sha256 when non-null). A mismatch is an "
                        "IDENTITY_MISMATCH -> content SKIP, not a FAIL.")
    args = p.parse_args()

    refs = referenced_models(args.workflow)
    print(f"referenced models: {len(refs)}")
    for filename, subdirs in refs:
        print(f"  - {filename}  (looked in: {', '.join(subdirs)})")

    if not args.comfyui_path:
        print("[info] no --comfyui_path given; cannot check presence.")
        sys.exit(0)

    missing = missing_models(args.workflow, args.comfyui_path)
    if missing:
        print(skip_message(missing))
        sys.exit(2)

    if args.verify:
        mismatches = identity_mismatches(args.workflow, args.comfyui_path)
        if mismatches:
            print("SKIP: model identity mismatch (wrong/corrupt file) - "
                  "re-provision:")
            for m in mismatches:
                print(f"  - {m}")
            sys.exit(2)

    print("all referenced models present"
          + (" and verified." if args.verify else "."))
    sys.exit(0)


if __name__ == "__main__":
    main()
