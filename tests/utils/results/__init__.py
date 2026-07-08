"""Results handling utilities (fleet-compatible subset for ComfyUI suite)."""

from .handler import ResultsHandler, build_artifactory_info
from .api import ResultsAPI, build_results_payload, validate_payload
from .rastra_link import (
    generate_rastra_link,
    generate_link_from_payload,
    append_to_github_summary,
    generate_and_append,
)

__all__ = [
    "ResultsHandler",
    "ResultsAPI",
    "build_results_payload",
    "validate_payload",
    "build_artifactory_info",
    "generate_rastra_link",
    "generate_link_from_payload",
    "append_to_github_summary",
    "generate_and_append",
]
