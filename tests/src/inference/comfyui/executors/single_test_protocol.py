#!/usr/bin/env python3
"""
Layer    : Execution
Equiv    : in-container test body of reusable-test-executor.yml (production)
Purpose  : Drives a single ComfyUI workflow through the ComfyUI REST API,
           measures warmup + timed runs, downloads generated output, performs
           a sanity check on the output type, and writes results.json with a
           PASS/FAIL verdict that comfyui_validator.py consumes.

ComfyUI REST endpoints used:
  GET  /system_stats   - health + env info
  POST /free           - flush VRAM between runs
  POST /interrupt      - stop the currently executing prompt
  POST /queue          - clear pending queue ({"clear": true})
  POST /prompt         - submit workflow JSON
  GET  /history/{id}   - poll completion + timing
  GET  /view           - download generated output

Exit codes:
  0  protocol completed (verdict written to results.json; PASS or FAIL inside)
  1  protocol could not run (workflow missing, server error, no output)
"""

import argparse, json, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_TYPE_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "video": {".mp4", ".webm", ".gif", ".mkv"},
    "audio": {".wav", ".flac", ".mp3", ".ogg"},
}


def parse_args():
    p = argparse.ArgumentParser(description="ComfyUI single test protocol")
    p.add_argument("--test_name",     required=True)
    p.add_argument("--workflow",      required=True)
    p.add_argument("--comfyui_url",   default="http://127.0.0.1:8188")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--warmup_runs",   type=int, default=0)
    p.add_argument("--timed_runs",    type=int, default=1)
    p.add_argument("--timeout_s",     type=int, default=600)
    p.add_argument("--expected_type", default="image")
    p.add_argument("--poll_interval", type=float, default=2.0)
    p.add_argument("--max_latency_s", type=float, default=None,
                   help="Perf gate: if the average timed-run latency exceeds this "
                        "many seconds the verdict is FAIL. Default None = no gate. "
                        "A slow pass is a FAIL, never a SKIP.")
    return p.parse_args()


def latency_gate(latency_avg_s, max_latency_s):
    """Pure perf-gate decision. Returns (result_str, failed_bool):
      * max_latency_s is None                 -> ("no_gate", False)
      * latency unknown (no timed successes)  -> ("no_latency", False)
      * avg <= max                            -> ("within", False)
      * avg  > max                            -> ("exceeded", True)
    A slow pass is a FAIL (True), never a SKIP - the caller must not downgrade it.
    """
    if max_latency_s is None:
        return "no_gate", False
    if latency_avg_s is None:
        return "no_latency", False
    if latency_avg_s > max_latency_s:
        return "exceeded", True
    return "within", False


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def format_http_error(code, reason, body):
    """Turn a ComfyUI error response into a readable message.

    On a rejected /prompt (HTTP 400) ComfyUI returns a JSON body of the form
    {"error": {...}, "node_errors": {"<id>": {"class_type": ..,
    "errors": [{"message": .., "details": ..}]}}} that names exactly which node
    failed validation (e.g. a LoadImage with a missing input image). The raw
    urllib HTTPError hides this body, so failures previously recorded only
    "HTTP Error 400: Bad Request". This surfaces the real cause."""
    text = ""
    try:
        text = (body.decode("utf-8", "replace")
                if isinstance(body, (bytes, bytearray)) else str(body or ""))
    except Exception:
        text = ""
    try:
        data = json.loads(text) if text else None
    except Exception:
        data = None

    parts = []
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            det = err.get("details")
            parts.append(f"{err['message']}: {det}".strip(": ").strip()
                         if det else err["message"])
        node_errors = data.get("node_errors")
        if isinstance(node_errors, dict):
            for nid, ninfo in node_errors.items():
                if not isinstance(ninfo, dict):
                    continue
                ctype = ninfo.get("class_type", "?")
                for e in (ninfo.get("errors") or []):
                    if not isinstance(e, dict):
                        continue
                    em, ed = e.get("message", ""), e.get("details", "")
                    detail = f"{em}: {ed}".strip(": ").strip() if ed else em
                    parts.append(f"{ctype}[{nid}] {detail}".strip())

    base = f"HTTP {code} {reason}".strip()
    if parts:
        return base + ": " + " | ".join(p for p in parts if p)
    snippet = " ".join(text.split())[:300]
    return f"{base}: {snippet}" if snippet else base


def http_post_json(url, payload, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        err_body = b""
        try:
            err_body = e.read()
        except Exception:
            pass
        raise RuntimeError(format_http_error(e.code, e.reason, err_body)) from None
    return json.loads(body) if body else {}


def free_vram(base_url):
    """Best-effort POST /free to flush VRAM between runs."""
    try:
        http_post_json(f"{base_url}/free", {"unload_models": True, "free_memory": True}, timeout=15)
    except Exception:
        pass


def cancel_running(base_url):
    """Best-effort stop of the currently executing prompt + clear of the pending
    queue. ComfyUI keeps sampling a submitted prompt even after our poller gives
    up; on the shared server that orphan blocks the next test's prompt (head-of-
    line blocking) and can push it past its own timeout. Call this on our own
    timeout and at the start of each test so no abandoned job survives."""
    for path, payload in (("/interrupt", {}), ("/queue", {"clear": True})):
        try:
            http_post_json(f"{base_url}{path}", payload, timeout=10)
        except Exception:
            pass


def load_workflow(path):
    """Load a ComfyUI workflow graph (API-format prompt dict)."""
    data = json.loads(Path(path).read_text())
    # ComfyUI accepts either a raw prompt mapping or {"prompt": {...}}.
    if "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data


def submit_prompt(base_url, prompt):
    resp = http_post_json(f"{base_url}/prompt", {"prompt": prompt})
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {resp}")
    return prompt_id


def wait_for_history(base_url, prompt_id, timeout_s, poll_interval):
    """Poll /history/{id} until the prompt completes or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            raw = http_get(f"{base_url}/history/{prompt_id}", timeout=10)
            hist = json.loads(raw)
        except Exception:
            hist = {}
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(poll_interval)
    # Stop the abandoned prompt + clear the queue so it cannot block later tests.
    cancel_running(base_url)
    raise TimeoutError(
        f"prompt {prompt_id} did not complete within {timeout_s}s "
        "(interrupted and queue cleared)")


def collect_outputs(history):
    """Extract output file descriptors from a history entry."""
    outputs = []
    for node_id, node_out in history.get("outputs", {}).items():
        for key in ("images", "gifs", "videos", "audio"):
            for item in node_out.get(key, []):
                if isinstance(item, dict) and item.get("filename"):
                    outputs.append(item)
    return outputs


def download_output(base_url, item, dest_dir):
    params = urllib.parse.urlencode({
        "filename":  item.get("filename", ""),
        "subfolder": item.get("subfolder", ""),
        "type":      item.get("type", "output"),
    })
    data = http_get(f"{base_url}/view?{params}", timeout=60)
    dest = Path(dest_dir) / item["filename"]
    dest.write_bytes(data)
    return dest


def sanity_check(files, expected_type):
    valid_ext = OUTPUT_TYPE_EXTENSIONS.get(expected_type, set())
    for f in files:
        if Path(f).suffix.lower() in valid_ext and Path(f).stat().st_size > 0:
            return True
    return False


def extract_exec_status(history):
    """Pull ComfyUI execution status + any node-level error messages from the
    /history entry, so failures record a real reason (not just 'no output')."""
    status = history.get("status", {}) if isinstance(history, dict) else {}
    status_str = status.get("status_str") or (
        "success" if status.get("completed") else "unknown")
    errors = []
    for msg in status.get("messages", []):
        try:
            ev, data = msg[0], msg[1]
        except Exception:
            continue
        if ev in ("execution_error", "execution_interrupted"):
            node = data.get("node_type") or data.get("node_id") or "?"
            detail = (data.get("exception_message")
                      or data.get("exception_type") or "")
            errors.append(f"{ev} @ {node}: {detail}".strip())
    return status_str, errors


# Environment/IO failures on the ComfyUI SAVE path (output dir not writable, disk
# full) are NOT model regressions - they are infrastructure problems. Detect them
# from error text so the caller can classify them as INFRA_ERROR, not FAIL.
_IO_ERROR_MARKERS = (
    "permission denied", "errno 13", "eacces",
    "operation not permitted", "errno 1", "eperm",
    "no space left", "errno 28", "enospc",
    "read-only file system", "errno 30", "erofs",
)


def is_io_error(text):
    """True if the error text looks like a filesystem/IO problem on the output
    path (EACCES/EPERM/ENOSPC/EROFS) rather than a model/content failure."""
    if not text:
        return False
    low = str(text).lower()
    return any(marker in low for marker in _IO_ERROR_MARKERS)


def classify_io_errors(errors):
    """Return the subset of error strings that are filesystem/IO failures."""
    return [e for e in (errors or []) if is_io_error(e)]


def run_once(base_url, prompt, timeout_s, poll_interval):
    start = time.perf_counter()
    prompt_id = submit_prompt(base_url, prompt)
    history = wait_for_history(base_url, prompt_id, timeout_s, poll_interval)
    elapsed = round(time.perf_counter() - start, 3)
    return prompt_id, history, elapsed


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "test_name":     args.test_name,
        "expected_type": args.expected_type,
        "comfyui_url":   args.comfyui_url,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "warmup_runs":   args.warmup_runs,
        "timed_runs":    args.timed_runs,
        "verdict":       "FAIL",
        "runs":          [],
        "errors":        [],
    }

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        result["errors"].append(f"workflow_not_found:{workflow_path}")
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        print(f"[FAIL] workflow not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    try:
        prompt = load_workflow(workflow_path)
    except Exception as e:
        result["errors"].append(f"workflow_parse_error:{e}")
        (out_dir / "results.json").write_text(json.dumps(result, indent=2))
        print(f"[FAIL] could not parse workflow: {e}", file=sys.stderr)
        sys.exit(1)

    base_url = args.comfyui_url.rstrip("/")

    # The ComfyUI server is shared across tests. If a previous test's poll timed
    # out, its prompt may still be running/queued here. Clear it first so this
    # test starts on an idle server and isn't blocked behind an orphan.
    cancel_running(base_url)

    # Warmup runs - not timed, results discarded.
    for i in range(args.warmup_runs):
        free_vram(base_url)
        try:
            run_once(base_url, prompt, args.timeout_s, args.poll_interval)
            print(f"[warmup {i + 1}/{args.warmup_runs}] ok")
        except Exception as e:
            result["errors"].append(f"warmup_{i + 1}_error:{e}")
            print(f"[warmup {i + 1}] error: {e}", file=sys.stderr)

    downloaded_files = []
    latencies = []

    for i in range(args.timed_runs):
        free_vram(base_url)
        try:
            prompt_id, history, elapsed = run_once(
                base_url, prompt, args.timeout_s, args.poll_interval)
        except Exception as e:
            result["errors"].append(f"timed_{i + 1}_error:{e}")
            result["runs"].append({"run": i + 1, "status": "ERROR", "detail": str(e)})
            print(f"[timed {i + 1}] error: {e}", file=sys.stderr)
            continue

        status_str, exec_errors = extract_exec_status(history)
        outputs = collect_outputs(history)
        run_files = []
        for item in outputs:
            try:
                dest = download_output(base_url, item, out_dir)
                run_files.append(str(dest))
            except Exception as e:
                result["errors"].append(f"download_error:{e}")

        downloaded_files.extend(run_files)
        latencies.append(elapsed)
        run_rec = {
            "run":            i + 1,
            "prompt_id":      prompt_id,
            "status":         "OK",
            "latency_s":      elapsed,
            "output_files":   run_files,
            "comfyui_status": status_str,
        }
        if exec_errors:
            run_rec["comfyui_errors"] = exec_errors
            result["errors"].extend(exec_errors)
        if not run_files:
            run_rec["note"] = "no_output_files_returned_by_comfyui"
        result["runs"].append(run_rec)
        extra = f" | comfyui={status_str}" + (f" | {exec_errors[0]}" if exec_errors else "")
        print(f"[timed {i + 1}/{args.timed_runs}] {elapsed}s, {len(run_files)} file(s){extra}")

    if latencies:
        result["latency_avg_s"] = round(sum(latencies) / len(latencies), 3)
        result["latency_min_s"] = round(min(latencies), 3)
        result["latency_max_s"] = round(max(latencies), 3)

    result["output_files"] = downloaded_files
    result["output_sanity_ok"] = sanity_check(downloaded_files, args.expected_type)

    has_timed_success = any(r.get("status") == "OK" for r in result["runs"])
    if has_timed_success and result["output_sanity_ok"]:
        result["verdict"] = "PASS"
    else:
        result["verdict"] = "FAIL"

    # Perf gate: a run that generated valid output but was too slow is a FAIL,
    # never a SKIP. Always record the gate outcome, even when no gate is set.
    latency_avg = result.get("latency_avg_s")
    gate_result, gate_failed = latency_gate(latency_avg, args.max_latency_s)
    result["latency_gate_s"] = args.max_latency_s
    result["latency_gate_result"] = gate_result
    if gate_failed:
        result["verdict"] = "FAIL"
        result["errors"].append(
            f"latency_gate: avg {latency_avg}s > max {args.max_latency_s}s")

    # Flag filesystem/IO failures (unwritable output dir, disk full) so the
    # caller can classify them as INFRA_ERROR, not a content FAIL. These come
    # through as node/execution errors from ComfyUI's save nodes.
    io_errs = classify_io_errors(result["errors"])
    result["io_error"] = bool(io_errs)
    if io_errs:
        result["io_error_detail"] = "; ".join(io_errs)

    # Record a human-readable failure reason so the log/report explains FAILs.
    if result["verdict"] == "FAIL":
        reasons = []
        run_errs = [r["detail"] for r in result["runs"]
                    if r.get("status") == "ERROR" and r.get("detail")]
        comfy_errs = [e for e in result["errors"]
                      if e.startswith("execution_")]
        if io_errs:
            reasons.append("filesystem/IO error on the ComfyUI output path "
                           "(environment, not a model regression): "
                           + "; ".join(io_errs))
        if run_errs:
            reasons.append("submit/poll error: " + "; ".join(run_errs))
        if comfy_errs:
            reasons.append("ComfyUI execution error: " + "; ".join(comfy_errs))
        if not downloaded_files and not run_errs:
            reasons.append("prompt completed but ComfyUI returned no output "
                           "files (likely a node/save error or unsupported "
                           "output node)")
        elif downloaded_files and not result["output_sanity_ok"]:
            reasons.append(f"output present but failed {args.expected_type} "
                           "sanity check (wrong type/empty file)")
        if gate_failed:
            reasons.append(f"latency gate: avg {latency_avg}s exceeds max "
                           f"{args.max_latency_s}s")
        result["failure_reason"] = " | ".join(reasons) or "unknown"

    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"[{result['verdict']}] {args.test_name}"
          + (f" -- reason: {result['failure_reason']}"
             if result.get("failure_reason") else ""))

    # Exit 0 means the protocol ran and produced a verdict file. The verdict
    # itself (PASS/FAIL) is read by comfyui_validator.py.
    sys.exit(0)


if __name__ == "__main__":
    main()
