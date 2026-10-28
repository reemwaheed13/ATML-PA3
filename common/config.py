"""YAML config loading and the shared --config / --resume CLI contract.

Every training/eval script takes --config (a YAML under configs/) and --resume, with nothing
hardcoded (handout: "preserve the configurations used to produce your results", Instructions
p1). This module loads YAML into plain dicts/dataclasses and builds the argparse parser so all
scripts expose the same flags.
"""

from __future__ import annotations

import argparse
from typing import Any


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    TODO: yaml.safe_load; optionally support a top-level `inherit:` key so smoke.yaml can
          override a base run config with a few fields.
    """
    raise NotImplementedError


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Standard parser: --config (required), --resume (flag), --drive-root (optional override).

    TODO: add arguments; --resume continues from the latest checkpoint for this config's run id.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--drive-root", default=None, help="Override DRIVE_ROOT (else env var).")
    return parser


def run_id_from_config(config_path: str, cfg: dict[str, Any]) -> str:
    """Derive a stable run identifier used to name checkpoint/eval subfolders on DRIVE_ROOT.

    TODO: prefer cfg["run_id"] if present, else the config filename stem.
    """
    raise NotImplementedError
