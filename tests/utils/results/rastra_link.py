"""
RASTRA Dashboard Link Generator - Framework Level Utility

Generates RASTRA dashboard links for any test that uploads results to the database.
Can be used by any test workflow to show clickable links in GitHub Actions UI.

Usage:
    # In Python script
    from tests.utils.results.rastra_link import generate_rastra_link, append_to_github_summary
    
    url = generate_rastra_link(
        rocm_version="6.4.0",
        execution_label="my-test-run"
    )
    append_to_github_summary(url)
    
    # CLI usage
    python -m tests.utils.results.rastra_link --rocm-version 6.4.0 --label my-test

    # From workflow YAML (after DB upload)
    python3 tests/utils/results/rastra_link.py --from-payload results/payload.json --append-summary
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode


# RASTRA Dashboard base URL
RASTRA_BASE_URL = "https://rastra-test-results.amd.com/performance-results"


def generate_rastra_link(
    rocm_version: str = None,
    rocm_release: str = None,
    execution_label: str = None,
    view_mode: str = "default",
    extra_params: dict = None
) -> str:
    """
    Generate RASTRA dashboard URL.
    
    Args:
        rocm_version: ROCm version (e.g., "6.4.0"). Auto-detected if not provided.
        rocm_release: ROCm release number (e.g., "6.4"). Derived from version if not provided.
        execution_label: Test execution label. Uses EXECUTION_LABEL env var if not provided.
        view_mode: Dashboard view mode (default: "default")
        extra_params: Additional URL parameters
        
    Returns:
        RASTRA dashboard URL
    """
    # Auto-detect rocm_version from environment if not provided
    if not rocm_version:
        rocm_version = os.environ.get("ROCM_VERSION", "")
    
    if not rocm_version:
        print("⚠ Warning: ROCm version not available, RASTRA link may show all versions")
    
    # Derive rocm_release from version if not provided
    # Example: "7.1.1-38" -> rocm_release="7.1.1", rocm_version="7.1.1-38"
    if not rocm_release and rocm_version:
        # Strip build number suffix (e.g., "7.1.1-38" -> "7.1.1")
        rocm_release = rocm_version.split("-")[0]
    
    # Get execution label from environment if not provided
    if not execution_label:
        execution_label = os.environ.get("EXECUTION_LABEL", "benchmark")
    
    # Build URL parameters
    params = {
        "rocmReleaseNumbers": rocm_release or "",
        "rocmVersions": rocm_version or "",
        "executionLabels": execution_label,
        "viewMode": view_mode
    }
    
    # Add extra parameters
    if extra_params:
        params.update(extra_params)
    
    # Remove empty parameters
    params = {k: v for k, v in params.items() if v}
    
    # Build URL
    url = f"{RASTRA_BASE_URL}?{urlencode(params)}"
    
    return url


def generate_link_from_payload(payload_path: str) -> str:
    """
    Generate RASTRA link by extracting info from payload.json.
    
    Args:
        payload_path: Path to payload.json file
        
    Returns:
        RASTRA dashboard URL
    """
    payload_file = Path(payload_path)
    
    if not payload_file.exists():
        print(f"[RASTRA] Warning: payload.json not found at {payload_path}")
        return ""
    
    try:
        with open(payload_file) as f:
            payload = json.load(f)
        
        # Extract ROCm version from payload
        # Try multiple locations: build_info (MAD), software_config (vLLM native)
        rocm_version = ""
        
        # Try build_info first (MAD upload format)
        build_info = payload.get("build_info", {})
        if build_info:
            rocm_version = build_info.get("rocm_version", "")
        
        # Fallback to software_config (vLLM native format)
        if not rocm_version:
            sw_config = payload.get("software_config", {})
            rocm_version = sw_config.get("rocm_version", "")
        
        if not rocm_version:
            print(f"[RASTRA] Warning: Could not find rocm_version in payload")
            return ""
        
        print(f"[RASTRA] Found ROCm version: {rocm_version}")
        
        # Get execution label from environment
        execution_label = os.environ.get("EXECUTION_LABEL", "benchmark")
        
        return generate_rastra_link(
            rocm_version=rocm_version,
            execution_label=execution_label
        )
        
    except Exception as e:
        print(f"[RASTRA] Error reading payload: {e}")
        return ""


def append_to_github_summary(
    url: str,
    title: str = "Results Dashboard",
    summary_file: str = None
) -> bool:
    """
    Append RASTRA link to GitHub Actions summary.
    
    Args:
        url: RASTRA dashboard URL
        title: Title for the link section
        summary_file: Optional file path to append to (also appends to GITHUB_STEP_SUMMARY)
        
    Returns:
        True if successfully appended
    """
    if not url:
        print("[RASTRA] No URL to append")
        return False
    
    # Build markdown content
    lines = [
        "",
        "---",
        f"## 📊 {title}",
        "",
        f"[Open Results in RASTRA Dashboard]({url})",
        "",
        "<details><summary>Direct Link</summary>",
        "",
        "```",
        url,
        "```",
        "</details>",
        ""
    ]
    
    content = "\n".join(lines)
    
    # Append to provided file
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write(content)
            print(f"[RASTRA] Link appended to {summary_file}")
        except Exception as e:
            print(f"[RASTRA] Error writing to {summary_file}: {e}")
    
    # Append to GitHub Actions summary
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        try:
            with open(github_summary, "a") as f:
                f.write(content)
            print(f"[RASTRA] Link appended to GitHub Actions summary")
            return True
        except Exception as e:
            print(f"[RASTRA] Error writing to GitHub summary: {e}")
            return False
    else:
        # Print to stdout for non-GitHub environments
        print(content)
        return True


def generate_and_append(
    rocm_version: str = None,
    execution_label: str = None,
    payload_path: str = None,
    summary_file: str = None,
    title: str = "Results Dashboard"
) -> str:
    """
    One-step function to generate RASTRA link and append to summary.
    
    Args:
        rocm_version: ROCm version (optional if payload_path provided)
        execution_label: Execution label
        payload_path: Path to payload.json (alternative to rocm_version)
        summary_file: Optional file to append to
        title: Title for the link section
        
    Returns:
        Generated URL
    """
    # Generate URL
    if payload_path:
        url = generate_link_from_payload(payload_path)
    else:
        url = generate_rastra_link(
            rocm_version=rocm_version,
            execution_label=execution_label
        )
    
    if url:
        append_to_github_summary(url, title=title, summary_file=summary_file)
        print(f"[RASTRA] Dashboard link: {url}")
    else:
        print("[RASTRA] Could not generate link - missing required info")
    
    return url


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate RASTRA dashboard link for test results"
    )
    
    parser.add_argument(
        "--rocm-version",
        help="ROCm version (e.g., 6.4.0)"
    )
    parser.add_argument(
        "--label", "--execution-label",
        dest="execution_label",
        help="Execution label for filtering results"
    )
    parser.add_argument(
        "--from-payload",
        dest="payload_path",
        help="Extract info from payload.json file"
    )
    parser.add_argument(
        "--append-summary",
        action="store_true",
        help="Append link to GitHub Actions summary"
    )
    parser.add_argument(
        "--output-file",
        help="Also append to this file"
    )
    parser.add_argument(
        "--title",
        default="Results Dashboard",
        help="Title for the link section"
    )
    
    args = parser.parse_args()
    
    # Generate URL
    if args.payload_path:
        url = generate_link_from_payload(args.payload_path)
    else:
        url = generate_rastra_link(
            rocm_version=args.rocm_version,
            execution_label=args.execution_label
        )
    
    if not url:
        print("[RASTRA] Failed to generate link")
        sys.exit(1)
    
    # Append to summary if requested
    if args.append_summary or args.output_file:
        append_to_github_summary(
            url,
            title=args.title,
            summary_file=args.output_file
        )
    else:
        # Just print the URL
        print(url)


if __name__ == "__main__":
    main()

