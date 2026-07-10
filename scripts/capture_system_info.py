#!/usr/bin/env python3
"""
Centralized Host/Container System Info Capture Script

Captures complete hardware configuration (GPU, CPU, ROCm) for database upload.
This is the SINGLE source of truth for bm_config.json across all test workflows.

Uses existing framework utilities:
- tests/utils/system/system_detector.py (SystemDetector)
- tests/utils/results/handler.py (ResultsHandler.build_system_info_dict)

Usage:
  # Capture host info (BEFORE Docker) - for bm_config
  python scripts/capture_system_info.py --output logs/bm_config.json
  
  # Capture container info (INSIDE Docker) - for container_info
  python scripts/capture_system_info.py --output logs/container_info.json
  
  # Include pip packages (slower, use inside Docker)
  python scripts/capture_system_info.py --output logs/container_info.json --packages
  
  # Capture full pip list to log file (for artifacts)
  python scripts/capture_system_info.py --output logs/container_info.json --pip-list

Used by: vLLM, MAD, JAX UT, and all functional test workflows.
"""

import argparse
import json
import sys
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force unbuffered output for CI logs
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


def capture_pip_list(output_dir: str) -> str:
    """
    Capture full pip list to a log file for artifacts.
    
    Works in any environment: Docker, venv, bare metal, wheel install.
    
    Args:
        output_dir: Directory to save pip_list.log
        
    Returns:
        Path to the saved log file
    """
    import subprocess
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "pip_list.log"
    
    print("  Capturing pip freeze...")
    
    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=60
        )
        
        # Build log content (pip freeze = pinned requirements format)
        content = [
            "=" * 70,
            "  PIP ENVIRONMENT LOG (pip freeze)",
            "=" * 70,
            f"Captured: {datetime.now().isoformat()}",
            f"Python: {sys.version}",
            f"Path: {sys.executable}",
            "",
            "=" * 70,
            "  PIP FREEZE (requirements format)",
            "=" * 70,
            pip_freeze.stdout if pip_freeze.returncode == 0 else f"Error: {pip_freeze.stderr}",
        ]
        
        with open(log_file, "w") as f:
            f.write("\n".join(content))
        
        pkg_count = len([l for l in pip_freeze.stdout.strip().split("\n") if l and "==" in l])
        print(f"  OK Pip freeze saved: {log_file} ({pkg_count} packages)")
        
        return str(log_file)
        
    except subprocess.TimeoutExpired:
        print("  WARN Pip freeze timed out")
        with open(log_file, "w") as f:
            f.write("ERROR: pip freeze command timed out after 60 seconds")
        return str(log_file)
    except Exception as e:
        print(f"  WARN Pip list failed: {e}")
        with open(log_file, "w") as f:
            f.write(f"ERROR: {e}")
        return str(log_file)


def capture_system_info(output_file: str, include_packages: bool = False, capture_pip: bool = False) -> dict:
    """
    Capture complete system info using existing framework utilities.
    
    Args:
        output_file: Path to save JSON output
        include_packages: Whether to detect key ML/AI packages for DB
        capture_pip: Whether to capture full pip list to log file
    
    Returns:
        dict: Complete system info
    """
    # Import framework utilities
    from tests.utils.system.system_detector import SystemDetector
    from tests.utils.results.handler import ResultsHandler
    
    print("=" * 60)
    print("  Capturing System Info")
    print("=" * 60)
    
    # Use existing SystemDetector (handles all OS/hardware detection)
    detector = SystemDetector()
    context = detector.detect_all(verbose=False)
    
    # Use existing ResultsHandler to build complete dict for DB
    result = ResultsHandler.build_system_info_dict(context)
    
    # Print summary
    print(f"  OS: {result.get('os', 'Unknown')}")
    print(f"  Kernel: {result.get('kernel', 'Unknown')}")
    print(f"  ROCm: {result.get('rocm_version', 'Unknown')}")
    print(f"  CPU: {result.get('cpu', {}).get('model', 'Unknown')}")
    print(f"  CPU Cores: {result.get('cpu', {}).get('cores', 0)}")
    print(f"  GPU: {result.get('gpu', {}).get('marketing_name', 'Unknown')}")
    print(f"  GPU Count: {result.get('gpu', {}).get('count', 0)}")
    print(f"  Partition: {result.get('gpu', {}).get('partition_mode', 'Unknown')}")
    
    # Add pip packages if requested (use existing utility)
    if include_packages:
        try:
            from tests.utils.common import get_environment_dependencies
            result["environment_dependencies"] = get_environment_dependencies()
            print(f"  Packages: {len(result.get('environment_dependencies', []))} detected")
        except ImportError:
            print("  Warning: Could not import get_environment_dependencies")
    
    # Capture full pip list to log file (for artifacts)
    if capture_pip:
        output_dir = Path(output_file).parent
        pip_log = capture_pip_list(str(output_dir))
        result["pip_list_log"] = pip_log
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print("")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Capture system info (GPU, CPU, ROCm) for database upload"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path (e.g., logs/bm_config.json)"
    )
    parser.add_argument(
        "--packages",
        action="store_true",
        help="Include pip package detection"
    )
    parser.add_argument(
        "--no-packages",
        action="store_true",
        help="Skip pip package detection (default)"
    )
    parser.add_argument(
        "--pip-list",
        action="store_true",
        help="Capture full pip list to pip_list.log (for artifacts)"
    )
    
    args = parser.parse_args()
    
    # --packages explicitly enables, --no-packages or default disables
    include_packages = args.packages and not args.no_packages
    capture_pip = args.pip_list
    
    try:
        capture_system_info(args.output, include_packages=include_packages, capture_pip=capture_pip)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
