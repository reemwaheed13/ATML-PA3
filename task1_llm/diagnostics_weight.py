"""Weight-space diagnostics: subspace-interference ratio rho_k and relative magnitude eta.
(handout p5, Step 6)

For the six checkpoints (Full FT @625 + @1250, L2-SP, LoRA-4, LoRA-16, OPLoRA @1250), on
down_proj in layers 0, 13, 27 (zero-based):

    rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2        (fraction of dW in the protected subspace)
    eta   = ||dW||_F / ||W0||_F                       (relative update magnitude)

U_k is the leading output-subspace basis of the starting weight W0 (the same k=128 bases OPLoRA
protects), from the SVD of W0 in FP32. dW is built by task1_llm.diagnostics_common (shared with
the activation diagnostics): Full-FT = W_trained - W0; LoRA = (alpha/r) B A; OPLoRA = (alpha/r)
B~ A~ with the projected factors. Everything FP32.

For OPLoRA we additionally verify the projection constraints numerically: ||U_k^T B~||_F and
||A~ V_k||_F, which must be ~0 if the orthogonal projection is correct. Results (one row per
checkpoint x layer) go to results/task1/diagnostics_weight.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Optional

import torch

from common.paths import RESULTS_DIR, ensure_dir

from . import diagnostics_common as dc
from .adapters import adapter_config_from_dict

log = logging.getLogger(__name__)


def leading_output_basis(w0: torch.Tensor, k: int) -> torch.Tensor:
    """Return U_k: the top-k left singular vectors of W0 (FP32)."""
    U_k, _V_k = dc.leading_bases(w0, k)
    return U_k


def rho_k(delta_w: torch.Tensor, u_k: torch.Tensor) -> float:
    """Return ||U_k U_k^T dW||_F^2 / ||dW||_F^2 (fraction of dW inside U_k's column space)."""
    proj = u_k @ (u_k.transpose(0, 1) @ delta_w)       # U_k U_k^T dW, via k-wide intermediate
    denom = float(delta_w.pow(2).sum())
    if denom == 0.0:
        return 0.0
    return float(proj.pow(2).sum()) / denom


def eta(delta_w: torch.Tensor, w0: torch.Tensor) -> float:
    """Return ||dW||_F / ||W0||_F."""
    w0_fro = float(torch.linalg.norm(w0))
    if w0_fro == 0.0:
        return 0.0
    return float(torch.linalg.norm(delta_w)) / w0_fro


def _oplora_constraints(train_cfg: dict[str, Any], weights: dict[str, Any],
                        w0: torch.Tensor, layer: int) -> Optional[dict[str, float]]:
    """For OPLoRA, report ||U_k^T B~||_F and ||A~ V_k||_F for `layer` (both ~0 if correct)."""
    if train_cfg.get("method") != "oplora":
        return None
    adapter_cfg = adapter_config_from_dict(train_cfg["adapter"])
    want = {layer}
    A = dc._collect_by_layer(weights, dc._DOWN_PROJ_A_RE, want)[layer]
    B = dc._collect_by_layer(weights, dc._DOWN_PROJ_B_RE, want)[layer]
    U_k, V_k = dc.leading_bases(w0, adapter_cfg.protect_k)
    A_tilde, B_tilde = dc.project_factors(A, B, U_k, V_k)
    return {
        "constraint_Uk_T_Btilde_fro": float(torch.linalg.norm(U_k.transpose(0, 1) @ B_tilde)),
        "constraint_Atilde_Vk_fro": float(torch.linalg.norm(A_tilde @ V_k)),
    }


def run(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Compute rho_k and eta for every (checkpoint, layer); write diagnostics_weight.json."""
    ws = cfg["weight_space"]
    layers = list(ws["layers"])
    k = int(ws["protect_k"])
    checkpoints = list(cfg["checkpoints"])
    if smoke:
        checkpoints, layers = dc.apply_smoke(cfg, checkpoints, layers)

    # W0 (original down_proj) + its leading U_k bases, computed once and reused across checkpoints.
    model_name = dc.base_model_name(dc.resolve_train_cfg(checkpoints[0]["run"]))
    log.info("Loading original weights for %s (layers %s).", model_name, layers)
    w0_by_layer = dc.load_original_down_proj(model_name, layers)
    uk_by_layer = {layer: leading_output_basis(w0_by_layer[layer], k) for layer in layers}

    rows: list[dict[str, Any]] = []
    for spec in checkpoints:
        run_id, update = spec["run"], int(spec["update"])
        train_cfg = dc.resolve_train_cfg(run_id, spec.get("config"))
        method = train_cfg["method"]
        log.info("Weight diagnostics: %s @%d (%s).", run_id, update, method)
        weights = dc.load_checkpoint_weights(run_id, update)
        dW_by_layer = dc.build_delta_weights(train_cfg, weights, w0_by_layer, layers)

        for layer in layers:
            dW, w0 = dW_by_layer[layer], w0_by_layer[layer]
            row = {
                "run": run_id, "update": update, "method": method, "layer": layer, "k": k,
                "rho_k": rho_k(dW, uk_by_layer[layer]),
                "eta": eta(dW, w0),
                "dW_fro": float(torch.linalg.norm(dW)),
                "W0_fro": float(torch.linalg.norm(w0)),
            }
            constraints = _oplora_constraints(train_cfg, weights, w0, layer)
            if constraints is not None:
                row.update(constraints)
            rows.append(row)
            log.info("  layer %2d: rho_k=%.4f eta=%.4g", layer, row["rho_k"], row["eta"])

    out = {
        "config": {"module": ws.get("module", "down_proj"), "layers": layers, "protect_k": k,
                   "compute_dtype": "float32", "smoke": smoke},
        "rows": rows,
    }
    out_dir = ensure_dir(RESULTS_DIR / "task1")
    out_path = out_dir / "diagnostics_weight.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[diag-weight] wrote {len(rows)} row(s) to {out_path}.", flush=True)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Task 1 weight-space diagnostics (Step 6).")
    parser.add_argument("--config", default="configs/task1/diagnostics.yaml",
                        help="Path to the diagnostics YAML config.")
    parser.add_argument("--drive-root", default=None, help="Override DRIVE_ROOT (else env var).")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast subset (one layer + first few checkpoints) for Colab testing.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root

    from common.config import load_yaml
    run(load_yaml(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
