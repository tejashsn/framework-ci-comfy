## Motivation

<!-- Why is this change needed? What problem does it solve?
     Reference prior PRs/issues/tickets where relevant, e.g.:
     - Fixes #123
     - Follow-up to #456
     - https://amd-hub.atlassian.net/browse/ROCM-XXXXX
-->

## Technical Details

<!-- How does the fix work? What files/flows changed?
     Call out constraints, version/platform scope, and runtime vs build-time behavior.
     For CI/suite changes: mention prefetch, validators, manifest, or runner impact. -->

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/tejashsn/framework-ci-comfy/blob/main/README.md
- [ ] `python -m pytest tests/unit -q` passes (no GPU required)
- [ ] If touching `suite_manifest.json`, regenerated `models_config.yaml` via `create_config.py --regenerate` when needed
- [ ] If adding workflow input assets, updated `config/workflow_inputs.json` and `assets/input/` (or ran `sync_workflow_inputs.py`)
- [ ] CI/workflow changes tested or dry-run on a self-hosted runner when applicable
