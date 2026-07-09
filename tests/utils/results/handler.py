"""Results handler for building, saving, and uploading test results."""

import os
import sys
import json
import time
import getpass
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime
from prettytable import PrettyTable
from decimal import Decimal, InvalidOperation

import logging
from .api import ResultsAPI, build_results_payload, validate_payload

if TYPE_CHECKING:
    from tests.test_core.base_test import TestResult

# Use standard logging
log = logging.getLogger(__name__)

# Constants
class Constants:
    TEST_ENV_BARE_METAL = "bm"
    TEST_ENV_VM = "vm"
    TEST_ENV_DOCKER = "bm-docker"

SEPARATOR_LINE = "=" * 70

# Artifactory defaults (can be overridden via env vars)
ARTIFACTORY_BASE_URL = "https://compute-artifactory.amd.com"
ARTIFACTORY_REPO_PATH = "artifactory/rocm-qa-test-logs/test-logs/framework-ci"
ARTIFACTORY_METADATA_DIR = "logs"

CONFIG_FILE_SEARCH_NAMES = [
    "system_snapshot.json",
    "bm_config.json",
    "host_bm_config.json",
    "container_info.json",
]

# Framework detection patterns: keyword → framework name
_FRAMEWORK_KEYWORDS = {
    "comfyui": "comfyui",
    "vllm": "vllm",
    "xdit": "xdit",
    "mad": "mad",
    "jax": "jax",
    "apex": "apex",
    "torch": "pytorch",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "triton": "triton",
}


def _slugify(value: str) -> str:
    """Convert a string to a safe path component (same as shell: tr '/' '-' | tr ':' '-')."""
    return value.replace("/", "-").replace(":", "-").replace(" ", "_")


def _load_config_data() -> dict:
    """Load system metadata from known config files in logs/ or workspace root."""
    search_dirs = [Path("logs"), Path(".")]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for name in CONFIG_FILE_SEARCH_NAMES:
            matches = sorted(search_dir.rglob(name))
            matches = [m for m in matches if ".git" not in m.parts]
            if matches:
                try:
                    with open(matches[0]) as f:
                        return json.load(f)
                except Exception:
                    continue
    return {}


def _extract_system_metadata(config_data: dict) -> dict:
    """Extract GPU, OS, ROCm from config data with platform-level fallbacks.

    Always returns a complete dict — never empty.  When config data is missing,
    uses Python's platform module and /opt/rocm as fallbacks.
    """
    host = config_data.get("host", {})
    container = config_data.get("container") or {}
    gpu_info = config_data.get("gpu", {}) or host.get("gpu", {})
    hardware = config_data.get("hardware", {})
    software = config_data.get("software", {})

    # --- GPU name ---
    gpu_name = (
        gpu_info.get("name")
        or hardware.get("gpu_name")
        or host.get("gpu", {}).get("name")
        or config_data.get("gpu_name")
        or os.environ.get("GPU_NAME", "")
        or os.environ.get("GPU_FAMILY", "")
        or "unknown_gpu"
    )
    gpu_name = _slugify(gpu_name)

    # --- GPU count ---
    gpu_count = (
        gpu_info.get("count")
        or hardware.get("gpu_count")
        or host.get("gpu", {}).get("count")
        or config_data.get("ngpu")
        or os.environ.get("GPU_COUNT", "")
        or os.environ.get("NGPU", "")
        or 0
    )
    if isinstance(gpu_count, str) and gpu_count.isdigit():
        gpu_count = int(gpu_count)

    # --- OS ---
    os_name = (
        config_data.get("os")
        or host.get("os")
        or config_data.get("os_name")
        or hardware.get("os")
        or os.environ.get("OS_NAME", "")
        or platform.system()
        or "UnknownOS"
    )
    os_version = (
        config_data.get("os_version", "")
        or host.get("os_version", "")
    )
    if not os_version:
        try:
            os_version = platform.version()
        except Exception:
            os_version = ""
    os_short = os_name.split()[0] if os_name else "UnknownOS"
    if os_version:
        ver_parts = os_version.split(".")
        os_short += ".".join(ver_parts[:2]) if len(ver_parts) >= 2 else os_version

    # --- ROCm version ---
    rocm_version = (
        container.get("rocm_version")
        or config_data.get("rocm_version")
        or host.get("rocm_version")
        or software.get("rocm_version")
        or config_data.get("build_info", {}).get("rocm_version")
        or os.environ.get("ROCM_VERSION", "")
        or ""
    )
    if not rocm_version:
        rocm_info = Path("/opt/rocm/.info/version")
        if rocm_info.exists():
            try:
                rocm_version = rocm_info.read_text().strip()
            except Exception:
                pass
    if not rocm_version:
        rocm_version = "unknown"
    rocm_short = rocm_version.split("-")[0] if "-" in rocm_version else rocm_version
    parts = rocm_short.split(".")
    if len(parts) > 3:
        rocm_short = ".".join(parts[:3])

    return {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "os_short": os_short,
        "rocm_version": rocm_short,
    }


def _derive_framework(test_name: str) -> str:
    """Derive framework name from env vars or test_name keywords."""
    fw = os.environ.get("FRAMEWORK_NAME", "")
    if fw:
        return fw.lower()

    name_lower = test_name.lower()
    for keyword, framework in _FRAMEWORK_KEYWORDS.items():
        if keyword in name_lower:
            return framework

    env_hints = {
        "VLLM_MODEL_NAME": "vllm",
        "XDIT_MODEL_NAME": "xdit",
        "MAD_MODEL": "mad",
        "FUNCTIONAL_CONFIG": None,  # checked below
    }
    for env_var, fw_name in env_hints.items():
        val = os.environ.get(env_var, "")
        if val:
            if fw_name:
                return fw_name
            for keyword, framework in _FRAMEWORK_KEYWORDS.items():
                if keyword in val.lower():
                    return framework

    return "unknown"


def _write_metadata_file(filename: str, content: str) -> None:
    """Write an Artifactory metadata file to logs/ directory."""
    try:
        meta_dir = Path(ARTIFACTORY_METADATA_DIR)
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / filename).write_text(content)
    except Exception:
        pass


def build_artifactory_info(test_name: str = "", framework: str = "") -> str:
    """Build the Artifactory URL for this test run and write the path to a file.

    This is the single source of truth for Artifactory path construction.
    Writes logs/artifactory_path.txt with the full upload path so the upload
    script (upload_to_artifactory.py) uses the exact same path — no mismatch.

    Args:
        test_name: Test identifier (typically result.test_name, e.g. "pyt_vllm_Llama-3.1-8B").
                   Slugified automatically.
        framework: Framework name. Auto-derived if empty.

    Returns:
        Full Artifactory URL string (never empty).
    """
    base_url = os.environ.get("ARTIFACTORY_BASE_URL", ARTIFACTORY_BASE_URL)
    repo_path = os.environ.get("ARTIFACTORY_REPO_PATH", ARTIFACTORY_REPO_PATH)

    config_data = _load_config_data()
    meta = _extract_system_metadata(config_data)

    # --- Framework ---
    if not framework:
        framework = _derive_framework(test_name)

    # --- Test name (slugified) ---
    if not test_name:
        test_name = os.environ.get("ARTIFACTORY_TEST_NAME", "")
    if not test_name:
        test_name = f"{framework}_test"
    test_name = _slugify(test_name)

    # --- Timestamp: reuse if already written, otherwise generate ---
    path_file = Path(ARTIFACTORY_METADATA_DIR) / "artifactory_path.txt"
    if path_file.exists():
        existing_path = path_file.read_text().strip()
        if existing_path:
            full_url = f"{base_url}{existing_path}"
            log.info(f"Artifactory URL (cached): {full_url}")
            return full_url

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- Build path ---
    dir_label = f"{meta['os_short']}_{meta['gpu_name']}_{meta['gpu_count']}-{test_name}"
    art_path = f"/{repo_path}/{framework}/{meta['rocm_version']}/{dir_label}/{timestamp}"
    full_url = f"{base_url}{art_path}"

    # Write the full path — upload script reads this as single source of truth
    _write_metadata_file("artifactory_path.txt", art_path)

    log.info(f"Artifactory URL: {full_url}")
    return full_url


class ResultsHandler:
    """Static methods for building, saving, and uploading test results."""

    @staticmethod
    def _get_node_count() -> int:
        """Get number of nodes from environment variables.
        
        Supports multiple formats:
        - MULTI_NODE_COUNT=4 (direct count)
        - RCCL_NODES=node1,node2,node3,node4 (comma-separated list)
        - NODE_LIST=node1,node2 (comma-separated list)
        
        Returns:
            int: Number of nodes (default 1 for single-node execution)
        """
        # Check for direct count
        if count := os.environ.get("MULTI_NODE_COUNT", "").strip():
            try:
                # Enforce minimum of 1 to ensure valid node count
                return max(1, int(count))
            except ValueError:
                pass
        
        # Check for node list (comma-separated)
        for env_var in ["RCCL_NODES", "NODE_LIST", "MPI_HOSTS"]:
            if nodes := os.environ.get(env_var, "").strip():
                node_list = [n.strip() for n in nodes.split(",") if n.strip()]
                if node_list:
                    return len(node_list)
        
        # Default to single node
        return 1

    @staticmethod
    def _extract_version(os_name: str) -> str:
        """Extract version number from OS name string.
        
        Args:
            os_name: OS name string (e.g., "Ubuntu 24.04.3 LTS", "Red Hat Enterprise Linux 8")
            
        Returns:
            str: Extracted version (e.g., "24.04.3", "8") or empty string
        """
        import re
        # Match version patterns: "8", "24.04", "24.04.3" (single digit or dotted versions)
        version_match = re.search(r'(\d+(?:\.\d+)*)', os_name)
        return version_match.group(1) if version_match else ""

    @staticmethod
    def build_deployment_info(
        config: Optional[Any] = None,
        deployed_by: str = "",
        execution_label: str = "",
    ) -> Dict[str, str]:
        """Build deployment information dict for test execution.

        Args:
            config: Configuration object (optional)
            deployed_by: Username who ran the test
            execution_label: Execution label

        Returns:
            Dict: Deployment info with timestamp, user, label, and command
        """

        # Build deployment info
        deployment_info = {
            "test_deployed_by": deployed_by,
            "test_deployed_on": datetime.now().isoformat(),
            "execution_label": execution_label,
            "test_flag": "prod_test",
            "testcase_command": " ".join(sys.argv),
            "execution_type": "automated",
        }

        log.debug(
            f"Deployment info: deployed_by={deployed_by}, command={deployment_info['testcase_command']}"
        )

        return deployment_info

    @staticmethod
    def build_system_info_dict(system_context: Any) -> Dict[str, Any]:
        """Build system info dict from SystemContext for API payload.

        Args:
            system_context: SystemContext object with detected system info

        Returns:
            Dict: System information (OS, CPU, GPU, ROCm details)
        """
        # Build OS string - avoid duplicate version if already in os_name (e.g., PRETTY_NAME)
        os_str = system_context.os_name or "Unknown"
        if system_context.os_version and system_context.os_version not in os_str:
            os_str = f"{os_str} {system_context.os_version}"
        
        # Get default IP if system_ip is None or "Unknown"
        system_ip = system_context.system_ip
        if not system_ip or system_ip == "Unknown":
            import socket
            try:
                system_ip = socket.gethostbyname(socket.gethostname()) or "0.0.0.0"
            except:
                system_ip = "0.0.0.0"
        
        # Flatten firmwares list to individual fields
        # Convert [{"name": "mec", "version": "41"}, ...] to {"mec": "41", ...}
        firmware_dict = {}
        for fw in (system_context.gpu_firmwares or []):
            if isinstance(fw, dict) and fw.get("name") and fw.get("version"):
                firmware_dict[fw["name"]] = fw["version"]
        
        # Build GPU dict with flattened firmwares
        gpu_info = {
            "count": system_context.gpu_count or 0,
            "name": system_context.gpu_name or "Unknown",
            "marketing_name": system_context.gpu_marketing_name or "Unknown",
            "device_id": system_context.gpu_device_id or "Unknown",
            "revision_id": system_context.gpu_revision_id or "Unknown",
            "vram_size": system_context.gpu_vram_size or 0,
            "sys_clock": system_context.gpu_sys_clock or 0,
            "mem_clock": system_context.gpu_mem_clock or 0,
            "vbios": system_context.gpu_vbios or "Unknown",
            "partition_mode": system_context.gpu_partition_mode or "Unknown",
            "xgmi_type": system_context.gpu_xgmi_type or "Unknown",
            "host_driver": system_context.gpu_host_driver or "Unknown",
            "no_of_nodes": ResultsHandler._get_node_count(),
            "devices": system_context.gpu_devices or [],
            "firmwares": system_context.gpu_firmwares or [],
            # Flattened firmware fields (kept for bm_config.json readability)
            "mec": firmware_dict.get("mec", ""),
            "mec2": firmware_dict.get("mec2", ""),
            "rlc": firmware_dict.get("rlc", ""),
            "sdma": firmware_dict.get("sdma", ""),
            "sdma2": firmware_dict.get("sdma2", ""),
            "vcn": firmware_dict.get("vcn", ""),
            "sos": firmware_dict.get("sos", ""),
            "ta_ras": firmware_dict.get("ta_ras", ""),
            "ta_xgmi": firmware_dict.get("ta_xgmi", ""),
            "smc": firmware_dict.get("smc", ""),
            "plfm_bundle": firmware_dict.get("plfm_bundle", ""),
        }
        
        return {
            "os": os_str.strip(),
            "os_version": system_context.os_version or ResultsHandler._extract_version(system_context.os_name or ""),
            "sbios": system_context.sbios or "Unknown",
            "kernel": system_context.kernel or "Unknown",
            "hostname": system_context.hostname or "Unknown",
            "system_ip": system_ip,
            "rocm_version": system_context.rocm_version or "Unknown",
            "cpu": {
                "model": system_context.cpu_model or "Unknown",
                "cores": system_context.cpu_cores or 0,
                "sockets": system_context.cpu_sockets or 1,
                "ram_size": system_context.cpu_ram_size or 0,
                "numa_nodes": system_context.cpu_numa_nodes or 1,
                "clock_speed": system_context.cpu_clock_speed or 0,
                "l1_cache": system_context.cpu_l1_cache or 0,
                "l2_cache": system_context.cpu_l2_cache or 0,
                "l3_cache": system_context.cpu_l3_cache or 0,
            },
            "gpu": gpu_info,
        }

    @staticmethod
    def build_docker_info_dict(
        container_info: Optional[Dict[str, Any]] = None,
        system_context: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build docker_info dict for API payload (docker_config section in DB).
        
        Can be built from either:
        - container_info: Dict loaded from container_info.json (captured inside Docker)
        - system_context: SystemContext from live detection (when running inside container)
        
        At least one source must be provided.
        
        Args:
            container_info: Dict with container info (os, os_version, cpu, gpu, hostname, etc.)
            system_context: SystemContext object from live detection
            
        Returns:
            Dict: Docker configuration info for DB, or None if no source provided
        """
        import os
        import socket
        
        if not container_info and not system_context:
            return None
        
        # Get values from container_info dict or system_context
        if container_info:
            container_os = container_info.get("os", "Unknown")
            container_os_version = container_info.get("os_version", "Unknown")
            cpu_info = container_info.get("cpu", {})
            cpu_cores = cpu_info.get("cores", 0)
            ram_gb = cpu_info.get("ram_size", cpu_info.get("ram_size_gb", 0))
            gpu_info = container_info.get("gpu", {})
            gpu_count = gpu_info.get("count", 0)
            container_hostname = container_info.get("hostname")
        else:
            container_os = system_context.os_name or "Unknown"
            container_os_version = system_context.os_version or "Unknown"
            cpu_cores = system_context.cpu_cores or 0
            ram_gb = system_context.cpu_ram_size or 0
            gpu_count = system_context.gpu_count or 0
            container_hostname = system_context.hostname
        
        # Keep RAM in GB — api.py's format_memory_size() expects GB
        cpu_ram_gb = ram_gb
        
        # Get hostname - prefer env var (runner name), then container_info, then socket
        hostname = os.environ.get("RUNNER_HOSTNAME") or container_hostname or socket.gethostname() or "Unknown"
        
        # Get IP address safely
        try:
            system_ip = socket.gethostbyname(socket.gethostname()) if socket.gethostname() else "127.0.0.1"
        except socket.gaierror:
            system_ip = "127.0.0.1"
        
        # Get cgroup info
        cgroup = "Unknown"
        try:
            with open("/sys/fs/cgroup/cgroup.controllers", "r") as f:
                cgroup = "v2"
        except FileNotFoundError:
            try:
                with open("/proc/self/cgroup", "r") as f:
                    content = f.read()
                    cgroup = "v2" if "0::/" in content else "v1"
            except Exception:
                pass
        
        return {
            "os": f"{container_os} {container_os_version}".strip(),
            "os_version": container_os_version,
            "hostname": hostname,
            "system_ip": system_ip,
            "docker_image": os.environ.get("DOCKER_IMAGE", ""),
            "base_docker_image": os.environ.get("BASE_DOCKER_IMAGE", "N/A"),
            "cpu": {
                "cores": cpu_cores,
                "ram_size": cpu_ram_gb,
            },
            "gpu": {
                "count": gpu_count,
            },
            "cgroup": cgroup,
        }

    @staticmethod
    def save_local_results(
        results_data: Dict[str, Any], output_dir: str, timestamp: Optional[str] = None
    ) -> Optional[Path]:
        """Save results to local JSON file with timestamp.

        Args:
            results_data: Results data dict
            output_dir: Output directory path
            timestamp: Optional timestamp (auto-generated if None)

        Returns:
            Path: Saved file path or None if failed
        """
        try:
            results_dir = Path(output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            if timestamp is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")

            output_file = results_dir / f"results_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2)

            log.info(f"✓ Results saved: {output_file}")
            return output_file
        except Exception as e:
            log.error(f"Failed to save local results: {e}")
            return None

    @staticmethod
    def upload_to_api(
        system_info: Dict[str, Any],
        test_results: List[Dict[str, Any]],
        timestamp: str,
        api_config: Dict[str, Any],
        rocm_info: Dict[str, Any],
        deployment_info: Dict[str, str],
        test_environment: str = Constants.TEST_ENV_BARE_METAL,
    ) -> bool:
        """Upload test results to API with system context.

        Args:
            system_info: System info dict
            test_results: List of test result dicts
            timestamp: Execution timestamp
            api_config: API config from ConfigHelper
            rocm_info: ROCm info dict
            deployment_info: Deployment info dict
            test_environment: Environment type (bm/vm/docker)

        Returns:
            bool: True if successful, False otherwise
        """
        # Check if API submission is enabled
        if not api_config.get("enabled", False):
            log.debug("API submission disabled")
            return False

        api_url = api_config.get("url", "")
        fallback_url = api_config.get("fallback_url", "")
        api_key = api_config.get("api_key", "")

        if not api_url:
            log.warning("API URL not configured, skipping submission")
            return False

        log.info(SEPARATOR_LINE)
        log.info("Submitting results to API...")
        log.info(SEPARATOR_LINE)

        try:
            # Build API payload
            test_results_for_api = []
            for result in test_results:
                test_results_for_api.append(
                    {
                        "test_name": result.get("test_name", ""),
                        "success": result.get("status", "PASS") in ("PASS", "PASSED")
                        or result.get("success", False),
                        "duration": result.get("duration", 0),
                        "error_message": result.get("error_message", ""),
                        "score": result.get("score"),  # Test score/value
                        "unit": result.get("unit", ""),  # Unit of measurement
                        "flag": result.get(
                            "flag", "H"
                        ),  # H (Higher is better) or L (Lower is better)
                        "test_config": result.get(
                            "test_config", {}
                        ),  # Test-specific configuration
                        "start_time": result.get(
                            "start_time", ""
                        ),  # Test start timestamp
                        "log_path": result.get("log_path", ""),  # Log file path
                    }
                )

            payload = build_results_payload(
                system_info=system_info,
                test_results=test_results_for_api,
                execution_time=timestamp,
                test_environment=test_environment,
                build_info=rocm_info,
                deployment_info=deployment_info,
            )

            log.debug(payload)

            # Validate payload
            if not validate_payload(payload):
                log.error("Payload validation failed, skipping API submission")
                return False

            # Submit to API
            api_client = ResultsAPI(api_url, api_key, fallback_url)
            success = api_client.submit_results(payload)

            if success:
                log.info("✓ Results submitted to API successfully")
                return True
            else:
                log.warning("⚠ Failed to submit results to API")
                return False

        except Exception as e:
            log.error(f"Unexpected error submitting to API: {e}")
            log.warning("Results not submitted - unexpected error")
            return False

    @staticmethod
    def fetch_lkg_scores_from_api(
        test_name: str, api_config: Dict[str, Any], rocm_info: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        """Fetch Last Known Good (LKG) scores from API for comparison.

        Args:
            test_name: Test name for LKG score lookup
            api_config: API config from ConfigHelper
            rocm_info: ROCm info dict

        Returns:
            Dict: Mapping of (test_name, sub_test_name) tuples to LKG scores
        """
        import requests  # Lazy import - only needed when fetching LKG scores
        
        try:
            # Get ROCm version
            rocm_version = rocm_info.get("rocm_version", "")

            # Validate API URL
            api_url = api_config.get("url", "")
            if not api_url:
                raise ValueError("API URL not configured in api_config.")

            # Build query string
            fetch_api_url = f"{api_url}/api/v1/rock-ci-results?skip=0&limit=600&rocm_version={rocm_version}&test_name={test_name}&lkg_score=true"

            # Make API request
            response = requests.get(
                fetch_api_url, headers={"accept": "application/json"}
            )
            response.raise_for_status()  # Raise error for HTTP issues
            data = response.json()

            # Validate response structure
            if "results" not in data:
                raise ValueError("Invalid API response: 'results' key not found.")

            # Build dictionary of LKG scores
            lkg_scores = {}
            for result in data.get("results", []):
                t_name = result.get("test_config", {}).get("test_name")
                sub_test_name = result.get("test_config", {}).get("sub_test_name")
                for metric in result.get("test_metrics", []):
                    lkg_score = metric.get("lkg_score")
                    if t_name and sub_test_name and lkg_score is not None:
                        lkg_scores[(t_name, sub_test_name)] = float(lkg_score)

            return lkg_scores

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Data error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    @staticmethod
    def get_final_result_table(
        table: PrettyTable, lkg_scores: Dict[Tuple[str, str], float]
    ) -> PrettyTable:
        """Augment PrettyTable with LKG comparison columns.

        Args:
            table: PrettyTable with test results
            lkg_scores: Mapping of (TestName, SubTests) to LKG scores

        Returns:
            PrettyTable: New table with LKGScores, %Diff, and FinalResult columns
        """

        # Validate required columns
        required_cols = ["TestName", "SubTests", "Scores", "Flag"]
        for col in required_cols:
            if col not in table.field_names:
                raise ValueError(f"Missing required column '{col}' in PrettyTable.")

        # Add new columns
        new_field_names = table.field_names + ["LKGScores", "%Diff", "FinalResult"]
        new_table = PrettyTable(new_field_names)

        for row in table._rows:  # Consider using table.rows if available
            row_dict = dict(zip(table.field_names, row))
            test_name = row_dict["TestName"]
            sub_test_name = row_dict["SubTests"]
            flag = row_dict["Flag"]

            try:
                score = Decimal(str(row_dict["Scores"]))
            except InvalidOperation:
                raise ValueError(f"Invalid score value: {row_dict['Scores']}")

            # Get LKG score
            lkg_score = lkg_scores.get((test_name, sub_test_name))
            diff: Optional[Decimal] = None
            final_result = "UNKNOWN"

            if lkg_score is not None:
                lkg_score_dec = Decimal(str(lkg_score))
                if flag == "H" and lkg_score_dec != 0:
                    diff = ((score - lkg_score_dec) / lkg_score_dec) * 100
                elif flag == "L" and lkg_score_dec != 0:
                    diff = ((lkg_score_dec - score) / lkg_score_dec) * 100

                # Determine FinalResult
                if diff is not None:
                    final_result = "FAIL" if diff < -5 else "PASS"

            # Append new values
            new_row = row + [
                float(lkg_score) if lkg_score is not None else None,
                round(float(diff), 2) if diff is not None else None,
                final_result,
            ]
            new_table.add_row(new_row)

        return new_table

    @staticmethod
    def build_failure_result(
        test_name: str,
        sub_test_name: str,
        log_url: str = "",
        python_version: str = "",
    ) -> Dict[str, Any]:
        """Build a failure result entry for DB upload."""
        if not python_version:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        try:
            from ..common import get_environment_dependencies
            env_deps = get_environment_dependencies()
        except Exception:
            env_deps = []
        
        return {
            "success": False,
            "duration": 0,
            "test_config": {
                "test_name": test_name,
                "sub_test_name": sub_test_name,
                "python_version": python_version,
                "environment_dependencies": env_deps,
            },
            "start_time": datetime.now().isoformat(),
            "log_path": log_url,
            "metrics": [],
        }

    @staticmethod
    def upload_failure(
        test_name: str,
        sub_test_name: str,
        log_url: str = "",
        execution_label: str = "",
        test_environment: str = Constants.TEST_ENV_DOCKER,
        system_context: Any = None,
        bm_config_path: Optional[Path] = None,
    ) -> bool:
        """Upload a failure result to the API.
        
        Used by workflows that don't use the test_runner framework (e.g., MAD).
        
        Args:
            test_name: Name of the test
            sub_test_name: Sub-test name
            log_url: URL to logs
            execution_label: Execution label for tracking
            test_environment: Test environment (bm-docker, etc.)
            system_context: Optional SystemContext (for live detection)
            bm_config_path: Optional path to bm_config.json (preferred over live detection)
        """
        import getpass
        
        system_info = None
        rocm_version = "Unknown"
        
        # PRIORITY 1: Try bm_config.json (same as upload_results)
        if bm_config_path is None:
            bm_config_path = ResultsHandler.find_config_file([
                Path("logs/bm_config.json"),
                Path("logs/host_bm_config.json"),
                Path("bm_config.json"),
            ])
        
        bm_config = ResultsHandler.load_json_file(bm_config_path)
        if bm_config:
            system_info = bm_config.copy()
            rocm_version = bm_config.get("rocm_version", "Unknown")
            log.info(f"Using bm_config.json for failure upload")
        
        # PRIORITY 2: Live detection (fallback)
        if system_info is None:
            if system_context is None:
                try:
                    from ..system import SystemDetector
                    detector = SystemDetector()
                    system_context = detector.detect_all(verbose=False)
                except Exception as e:
                    log.warning(f"Could not detect system info: {e}")
                    return False
            
            system_info = ResultsHandler.build_system_info_dict(system_context)
            rocm_version = system_context.rocm_version or "Unknown"
        
        deployment_info = ResultsHandler.build_deployment_info(
            deployed_by=getpass.getuser(),
            execution_label=execution_label,
        )
        
        rocm_info = {
            "rocm_version": rocm_version,
            "rocm_build_type": getattr(system_context, 'rocm_build_type', "Unknown") if system_context else "Unknown",
            "rocm_build_lib_type": getattr(system_context, 'rocm_build_lib_type', "Unknown") if system_context else "Unknown",
            "rocm_package_manager": getattr(system_context, 'rocm_package_manager', "Unknown") if system_context else "Unknown",
            "rocm_package_manager_version": getattr(system_context, 'rocm_package_manager_version', "Unknown") if system_context else "Unknown",
            "install_type": getattr(system_context, 'install_type', "Unknown") if system_context else "Unknown",
        }
        
        # Prefer Artifactory URL if available, else use provided log_url
        effective_log_url = log_url
        try:
            art_url = build_artifactory_info(test_name=test_name)
            if art_url:
                effective_log_url = art_url
        except Exception:
            pass

        failure_result = ResultsHandler.build_failure_result(
            test_name=test_name,
            sub_test_name=sub_test_name,
            log_url=effective_log_url,
        )
        
        payload = build_results_payload(
            system_info=system_info,
            test_results=[failure_result],
            execution_time=datetime.now().isoformat(),
            test_environment=test_environment,
            build_info=rocm_info,
            deployment_info=deployment_info,
        )
        
        if not validate_payload(payload):
            log.error("Failure payload validation failed")
            return False
        
        api_url = os.environ.get("API_URL") or "http://api.rastra-test-results.amd.com/api/v2/perf/rastra-test-results"
        api_key = os.environ.get("API_TOKEN") or ""
        
        api_client = ResultsAPI(api_url, api_key)
        success = api_client.submit_results(payload)
        
        if success:
            log.info(f"✓ Failure result uploaded: {test_name}/{sub_test_name}")
        else:
            log.warning(f"⚠ Failed to upload failure result: {test_name}/{sub_test_name}")
        
        return success

    # ==================== NEW SHARED UTILITIES ====================
    # These methods centralize common functionality for all tests
    # (vLLM, MAD, JAX, xDiT, etc.) to use

    @staticmethod
    def detect_environment() -> str:
        """Auto-detect if running inside Docker container or on bare metal.
        
        Detection methods (in order of reliability):
        1. Check /.dockerenv file (Docker creates this)
        2. Check /run/.containerenv (Podman creates this)
        3. Check /proc/1/cgroup for container indicators
        
        Returns:
            str: "bm-docker" if inside container, "bm" if bare metal
            Valid values: ['bm', 'vm', 'bm-docker', 'vm-docker']
        """
        from pathlib import Path
        
        # Method 1: Docker creates this file
        if Path("/.dockerenv").exists():
            return "bm-docker"
        
        # Method 2: Podman creates this file
        if Path("/run/.containerenv").exists():
            return "bm-docker"
        
        # Method 3: Check cgroup for container indicators
        try:
            cgroup_path = Path("/proc/1/cgroup")
            if cgroup_path.exists():
                cgroup_content = cgroup_path.read_text()
                if any(x in cgroup_content for x in ["docker", "containerd", "lxc", "kubepods"]):
                    return "bm-docker"
        except (PermissionError, OSError):
            pass
        
        # Default: bare metal (API expects "bm", not "bm-host")
        return "bm"

    @staticmethod
    def load_json_file(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        """Load JSON file with error handling.
        
        Args:
            path: Path to JSON file (can be None)
            
        Returns:
            Dict if successful, None if file doesn't exist or fails to parse
        """
        if path is None:
            return None
        
        if not path.exists():
            return None
        
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load JSON file {path}: {e}")
            return None

    @staticmethod
    def find_config_file(candidates: List[Path]) -> Optional[Path]:
        """Find first existing file from list of candidates.
        
        Args:
            candidates: List of potential file paths to check
            
        Returns:
            First existing path, or None if none exist
        """
        for path in candidates:
            if path and path.exists():
                return path
        return None

    @staticmethod
    def build_rocm_info(
        system_ctx: Optional[Any] = None,
        container_info: Optional[Dict[str, Any]] = None,
        bm_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build ROCm info dict from available sources.
        
        Priority: container_info > bm_config > system_ctx > defaults
        
        Args:
            system_ctx: SystemContext from live detection
            container_info: Dict from container_info.json
            bm_config: Dict from host_bm_config.json
            
        Returns:
            Dict with ROCm version and build info
        """
        if container_info:
            return {
                "rocm_version": container_info.get("rocm_version") or "Unknown",
                "rocm_build_type": container_info.get("rocm_build_type") or "Unknown",
                "rocm_build_lib_type": container_info.get("rocm_build_lib_type") or "Unknown",
                "rocm_package_manager": container_info.get("rocm_package_manager") or "Unknown",
                "rocm_package_manager_version": container_info.get("rocm_package_manager_version") or "Unknown",
                "install_type": container_info.get("install_type") or "Unknown",
            }
        
        if bm_config:
            return {
                "rocm_version": bm_config.get("rocm_version") or "Unknown",
                "rocm_build_type": bm_config.get("rocm_build_type") or "Unknown",
                "rocm_build_lib_type": bm_config.get("rocm_build_lib_type") or "Unknown",
                "rocm_package_manager": bm_config.get("rocm_package_manager") or "Unknown",
                "rocm_package_manager_version": bm_config.get("rocm_package_manager_version") or "Unknown",
                "install_type": bm_config.get("install_type") or "Unknown",
            }
        
        if system_ctx:
            return {
                "rocm_version": getattr(system_ctx, 'rocm_version', None) or "Unknown",
                "rocm_build_type": getattr(system_ctx, 'rocm_build_type', None) or "Unknown",
                "rocm_build_lib_type": getattr(system_ctx, 'rocm_build_lib_type', None) or "Unknown",
                "rocm_package_manager": getattr(system_ctx, 'rocm_package_manager', None) or "Unknown",
                "rocm_package_manager_version": getattr(system_ctx, 'rocm_package_manager_version', None) or "Unknown",
                "install_type": getattr(system_ctx, 'rocm_install_type', None) or "Unknown",
            }
        
        return {
            "rocm_version": "Unknown",
            "rocm_build_type": "Unknown",
            "rocm_build_lib_type": "Unknown",
            "rocm_package_manager": "Unknown",
            "rocm_package_manager_version": "Unknown",
            "install_type": "Unknown",
        }

    @staticmethod
    def convert_results_to_api_format(
        results: List["TestResult"],
        env_deps: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert TestResult objects to API payload format.
        
        Creates one API entry per metric (as required by DB schema).
        
        Args:
            results: List of TestResult objects
            env_deps: Environment dependencies (auto-detected if not provided)
            
        Returns:
            List of dicts ready for API payload
        """
        from ..common import get_environment_dependencies
        
        # Auto-detect environment dependencies if not provided
        if env_deps is None:
            env_deps = get_environment_dependencies()
        
        test_results_for_api = []
        
        # Build Artifactory URL once for all results (same run = same URL)
        # Uses first result's test_name for path derivation
        _artifactory_url_cache = ""
        if results:
            first_name = results[0].test_name if results else ""
            _artifactory_url_cache = build_artifactory_info(test_name=first_name)
        
        for result in results:
            # Priority: Artifactory URL > metadata log_file > GitHub ARTIFACT_URL
            log_path = ""
            if _artifactory_url_cache:
                log_path = _artifactory_url_cache
            elif result.metadata and result.metadata.get("log_file"):
                log_path = result.metadata.get("log_file")
            elif os.environ.get("ARTIFACT_URL"):
                log_path = os.environ.get("ARTIFACT_URL")
            
            # Build base test_config
            base_test_config = {
                "test_name": result.test_name,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "environment_dependencies": env_deps,
            }
            
            # Add test-specific parameters
            if result.metadata and "test_parameters" in result.metadata:
                test_params = result.metadata["test_parameters"]
                if isinstance(test_params, dict):
                    for key, value in test_params.items():
                        base_test_config[key] = value
            
            # Create ONE RESULT ENTRY PER METRIC
            if result.metrics:
                for name, data in result.metrics.items():
                    unit = data.get("unit") or ""
                    score = float(data.get("value", 0))
                    
                    # Determine flag (H=higher better, L=lower better)
                    if data.get("flag"):
                        flag = data.get("flag")
                    elif unit and unit.lower() in ["ms", "s", "seconds", "milliseconds", "us", "microseconds"]:
                        flag = "L"
                    elif any(x in name.lower() for x in ["tpot", "ttft", "itl", "e2el", "latency", "elapsed", "time"]):
                        flag = "L"
                    else:
                        flag = "H"
                    
                    metric_test_config = base_test_config.copy()
                    # Use adapter-set sub_test_name if available, else construct from test_name/test_type
                    metric_test_config["sub_test_name"] = getattr(result, 'sub_test_name', None) or f"{result.test_name}_{result.test_type}"
                    
                    # Only add metric to test_config if explicitly requested
                    if result.metadata and result.metadata.get("include_metric_in_test_config", False):
                        metric_test_config["metric"] = name
                    
                    test_results_for_api.append({
                        "success": result.status == "PASSED",
                        "duration": result.duration,
                        "error_message": result.error_message or "",
                        "test_config": metric_test_config,
                        "start_time": result.start_time.isoformat() if result.start_time else datetime.now().isoformat(),
                        "log_path": str(log_path) if log_path else "",
                        "metrics": [{
                            "metric_name": name,
                            "score": score,
                            "unit": unit,
                            "flag": flag,
                            "primary": True
                        }],
                    })
            else:
                # No metrics - handle based on status
                test_config = base_test_config.copy()
                test_config["sub_test_name"] = getattr(result, 'sub_test_name', result.test_type)
                
                if result.status == "PASSED":
                    test_results_for_api.append({
                        "success": True,
                        "duration": result.duration,
                        "error_message": "",
                        "test_config": test_config,
                        "start_time": result.start_time.isoformat() if result.start_time else datetime.now().isoformat(),
                        "log_path": str(log_path) if log_path else "",
                        "metrics": [{
                            "metric_name": "execution_time",
                            "score": result.duration,
                            "unit": "seconds",
                            "flag": "L",
                            "primary": True
                        }],
                    })
                elif result.status in ("FAILED", "ERROR"):
                    test_results_for_api.append({
                        "success": False,
                        "duration": result.duration,
                        "error_message": result.error_message or "",
                        "test_config": test_config,
                        "start_time": result.start_time.isoformat() if result.start_time else datetime.now().isoformat(),
                        "log_path": str(log_path) if log_path else "",
                        "metrics": [],
                    })
                elif result.status == "SKIPPED":
                    test_config["skip_reason"] = result.error_message or "Skipped"
                    test_results_for_api.append({
                        "success": True,  # Skipped is not a failure
                        "duration": 0,
                        "error_message": result.error_message or "Skipped",
                        "test_config": test_config,
                        "start_time": result.start_time.isoformat() if result.start_time else datetime.now().isoformat(),
                        "log_path": str(log_path) if log_path else "",
                        "metrics": [],
                    })
        
        return test_results_for_api

    @staticmethod
    def upload_results(
        results: List["TestResult"],
        bm_config_path: Optional[Path] = None,
        container_info_path: Optional[Path] = None,
        execution_label: str = "",
        test_environment: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """Upload test results to database - ONE-STOP function for any test.
        
        This is the main entry point for uploading results. It handles:
        - Auto-detection of environment (Docker vs bare metal)
        - Auto-finding config files if not specified
        - Building all required payloads (system_info, docker_info, rocm_info)
        - Converting TestResult objects to API format
        - Validation and upload
        
        Args:
            results: List of TestResult objects (from any test/adapter)
            bm_config_path: Path to host bm_config.json (auto-found if None)
            container_info_path: Path to container_info.json (auto-found if None)
            execution_label: Execution label for tracking
            test_environment: Environment type (auto-detected if None)
            api_url: API endpoint (from env if None)
            api_key: API key (from env if None)
            
        Returns:
            True if upload successful, False otherwise
            
        Example:
            # Simplest usage - framework handles everything
            ResultsHandler.upload_results(results)
            
            # With execution label
            ResultsHandler.upload_results(results, execution_label="nightly")
            
            # With explicit paths
            ResultsHandler.upload_results(
                results,
                bm_config_path=Path("logs/host_bm_config.json"),
                container_info_path=Path("logs/container_info.json"),
            )
        """
        if not results:
            log.warning("No results to upload")
            return True
        
        log.info(f"📤 Uploading {len(results)} result(s) to database...")
        
        # Step 1: Auto-detect environment if not specified
        if test_environment is None:
            test_environment = ResultsHandler.detect_environment()
            log.info(f"🔍 Auto-detected environment: {test_environment}")
        else:
            log.info(f"📍 Environment: {test_environment}")
        
        # Step 2: Load system info — prefer system_snapshot.json (single source),
        # fall back to separate bm_config.json + container_info.json for backward compat.
        bm_config = None
        container_info = None
        
        snapshot = ResultsHandler.load_json_file(Path("logs/system_snapshot.json"))
        if snapshot and snapshot.get("host"):
            bm_config = snapshot["host"]
            container_info = snapshot.get("container")  # None for non-Docker
            log.info("📋 Loaded system info from: logs/system_snapshot.json")
        else:
            # Backward compatibility: fall back to separate files
            if bm_config_path is None:
                bm_config_path = ResultsHandler.find_config_file([
                    Path("logs/bm_config.json"),
                    Path("logs/host_bm_config.json"),
                    Path("bm_config.json"),
                ])
            if container_info_path is None and test_environment == "bm-docker":
                container_info_path = ResultsHandler.find_config_file([
                    Path("logs/container_info.json"),
                    Path("container_info.json"),
                ])
            bm_config = ResultsHandler.load_json_file(bm_config_path)
            container_info = ResultsHandler.load_json_file(container_info_path)
        
        if bm_config:
            log.info(f"🖥️ Host: {bm_config.get('gpu', {}).get('marketing_name', 'Unknown')}")
        if container_info:
            log.info(f"📦 Container: ROCm {container_info.get('rocm_version', 'Unknown')}")
        
        # Step 4: Build system_info (from bm_config or live detection)
        system_info = None
        system_ctx = None
        
        if bm_config:
            # Use bm_config directly - it's already in correct format from
            # capture_system_info.py which uses build_system_info_dict()
            system_info = bm_config.copy()
            
            # Merge GPU firmware from container if host firmware is empty.
            # Host capture runs before Docker (amd-smi may not be installed),
            # but container detection runs inside Docker where amd-smi works.
            host_fw = system_info.get("gpu", {}).get("firmwares", [])
            if not host_fw and container_info:
                container_fw = container_info.get("gpu", {}).get("firmwares", [])
                if container_fw:
                    if "gpu" not in system_info:
                        system_info["gpu"] = {}
                    system_info["gpu"]["firmwares"] = container_fw
                    log.info(f"   Firmware: merged {len(container_fw)} entries from container")
            
            # Ensure system_ip has valid value
            if not system_info.get("system_ip") or system_info.get("system_ip") == "Unknown":
                import socket
                try:
                    system_info["system_ip"] = socket.gethostbyname(socket.gethostname()) or "0.0.0.0"
                except:
                    system_info["system_ip"] = "0.0.0.0"
            
            gpu_info = system_info.get("gpu", {})
            log.info(f"   GPU: {gpu_info.get('marketing_name', 'Unknown')} x{gpu_info.get('count', 0)}")
            log.info(f"   Partition: {gpu_info.get('partition_mode', 'Unknown')}")
        else:
            # Fallback: live detection
            log.info("📊 Detecting system info (no bm_config found)...")
            try:
                from ..system import SystemDetector
                detector = SystemDetector()
                system_ctx = detector.detect_all(verbose=False)
                system_info = ResultsHandler.build_system_info_dict(system_ctx)
            except Exception as e:
                log.error(f"System detection failed: {e}")
                return False
        
        # Step 5: Build docker_info (only if in Docker)
        docker_info = None
        if test_environment == "bm-docker":
            if container_info:
                docker_info = ResultsHandler.build_docker_info_dict(container_info=container_info)
            elif system_ctx:
                docker_info = ResultsHandler.build_docker_info_dict(system_context=system_ctx)
            
            if docker_info:
                log.info(f"🐳 Docker: {docker_info['os']}, {docker_info['cpu']['cores']} cores")
        
        # Step 6: Build rocm_info
        rocm_info = ResultsHandler.build_rocm_info(
            system_ctx=system_ctx,
            container_info=container_info,
            bm_config=bm_config,
        )
        log.info(f"🔧 ROCm: {rocm_info['rocm_version']}")
        
        # Step 7: Build deployment_info
        deployment_info = ResultsHandler.build_deployment_info(
            deployed_by=getpass.getuser(),
            execution_label=execution_label or os.environ.get("EXECUTION_LABEL", ""),
        )
        
        # Step 8: Get environment dependencies
        env_deps = None
        if container_info and container_info.get("environment_dependencies"):
            env_deps = container_info["environment_dependencies"]
        
        # Step 9: Convert results to API format
        test_results_for_api = ResultsHandler.convert_results_to_api_format(
            results=results,
            env_deps=env_deps,
        )
        
        if not test_results_for_api:
            log.warning("No results to upload after conversion")
            return True
        
        log.info(f"   Prepared {len(test_results_for_api)} metric entries")
        
        # Step 10: Build payload
        payload = build_results_payload(
            system_info=system_info,
            test_results=test_results_for_api,
            execution_time=datetime.now().isoformat(),
            test_environment=test_environment,
            build_info=rocm_info,
            deployment_info=deployment_info,
            docker_info=docker_info,
        )
        
        # Step 11: Validate payload
        if not validate_payload(payload):
            log.error("❌ Payload validation failed")
            try:
                payload_file = Path("logs/failed_payload.json")
                payload_file.parent.mkdir(parents=True, exist_ok=True)
                with open(payload_file, "w") as f:
                    json.dump(payload, f, indent=2, default=str)
                log.error(f"   Saved failed payload to: {payload_file}")
            except Exception:
                pass
            return False
        
        log.info("✓ Payload validated")
        
        # Save payload for debugging
        try:
            payload_file = Path("logs/payload.json")
            payload_file.parent.mkdir(parents=True, exist_ok=True)
            with open(payload_file, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception:
            pass
        
        # Step 12: Get API config
        if api_url is None:
            api_url = os.environ.get("API_URL") or "http://api.rastra-test-results.amd.com/api/v2/perf/rastra-test-results"
        if api_key is None:
            api_key = os.environ.get("API_TOKEN") or ""
        
        # Step 13: Upload
        api_client = ResultsAPI(api_url, api_key)
        success = api_client.submit_results(payload)
        
        if success:
            log.info(f"✅ Uploaded {len(test_results_for_api)} result(s) to database")
        else:
            log.error("❌ Upload failed")
        
        return success
