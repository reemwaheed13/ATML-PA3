"""YAML config loading and the shared --config / --resume CLI contract.

Every training/eval script takes --config (a YAML under configs/) and --resume, with nothing
hardcoded (handout: "preserve the configurations used to produce your results", Instructions
p1). This module loads YAML into plain dicts/dataclasses and builds the argparse parser so all
scripts expose the same flags.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base` (override wins); nested dicts are merged.

    This is what lets a per-run config override a single nested field (e.g. smoke.yaml setting
    only `data.num_examples`) without dropping the sibling keys it inherits.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML config, resolving a top-level `inherit:` chain.

    A config may set `inherit: <relative path>` (resolved against the config's own directory) to
    build on another config; the child is deep-merged over the resolved parent. Inheritance
    chains (smoke.yaml -> lora_r16.yaml -> _base.yaml) are resolved recursively. The `inherit`
    key itself is stripped from the returned dict.
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherit = cfg.pop("inherit", None)
    if inherit:
        parent_path = os.path.join(os.path.dirname(path), inherit)
        cfg = _deep_merge(load_yaml(parent_path), cfg)
    return cfg


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Standard parser: --config (required), --resume (flag), --drive-root + --stop-after.

    --resume continues from the latest checkpoint for this config's run id. --stop-after N exits
    cleanly after N optimizer updates (used to simulate a Colab disconnect for resume testing).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--drive-root", default=None, help="Override DRIVE_ROOT (else env var).")
    parser.add_argument(
        "--stop-after", type=int, default=None, metavar="N",
        help="Exit cleanly after N optimizer updates (after any checkpoint at N); "
             "simulates a Colab disconnect for testing --resume.",
    )
    return parser


def run_id_from_config(config_path: str, cfg: dict[str, Any]) -> str:
    """Derive a stable run identifier used to name checkpoint/eval subfolders on DRIVE_ROOT.

    Prefers an explicit `run_id` in the config; otherwise falls back to the config filename stem.
    """
    run_id = cfg.get("run_id")
    if run_id:
        return str(run_id)
    return os.path.splitext(os.path.basename(os.fspath(config_path)))[0]
