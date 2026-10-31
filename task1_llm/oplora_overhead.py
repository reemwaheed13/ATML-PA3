"""OPLoRA overhead accounting: basis storage, preprocessing time, export overhead. (handout Step 4)

Step 4 requires recording, for OPLoRA, the extra cost its orthogonal projection adds over plain
LoRA — without retraining. Three quantities, all measured on the real Qwen3-0.6B backbone loaded
exactly as OPLoRA training loads it (FP16, frozen):

  * Preprocessing time — the one-off cost of computing the leading bases U_k, V_k (k=128) via an
    FP32 SVD of W0 for every targeted module (q/k/v/o/gate/up/down in all 28 layers). This is the
    SVD done once in adapters.LoRAOPLoRALinear.__init__; here we time it standalone with
    torch.cuda.synchronize around the timer so the GPU work is actually captured.
  * Basis storage — total bytes of all U_k + V_k held in FP32 (the buffers OPLoRA must keep that
    LoRA does not).
  * Export overhead — the cost of turning the final checkpoint's A, B into projected effective
    factors and merging them into the backbone, versus the SAME merge for LoRA-16 with no
    projection. The projection step (A~, B~) is isolated from the deep-copy+add that both methods
    share, so the delta is attributable to OPLoRA's projection.

Reuses the training/eval code paths (model.load_base_model, model.configure_method,
adapters.merge_into_backbone, evaluate._load_trained_weights) so the numbers reflect the real
build, not a reimplementation. Writes results/task1/oplora_overhead.json.

--smoke restricts every per-module loop to the first 2 transformer layers (a quick wiring check).
The export timing depends only on tensor shapes, not values, so if a trained checkpoint is absent
(e.g. a local smoke run) the freshly-initialized factors are used and the timing is still
representative; whether trained weights were loaded is recorded per method.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from common.checkpoint import checkpoint_path
from common.config import build_arg_parser, load_yaml, run_id_from_config
from common.paths import RESULTS_DIR, ensure_dir, task1_dir

from .adapters import merge_into_backbone
from .evaluate import _load_trained_weights
from .model import configure_method, load_base_model

log = logging.getLogger(__name__)

_SMOKE_LAYERS = 2
_FINAL_UPDATE = 1250          # Step-4 final checkpoint (all adapter runs eval at 1250)


def _sync(device) -> None:
    """Block until queued GPU work finishes, so wall-clock timers capture it (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _in_first_layers(name: str, n_layers: int) -> bool:
    """True if a module path belongs to transformer layer 0..n_layers-1 (for --smoke)."""
    return any(f".layers.{i}." in name for i in range(n_layers))


def _targeted_linears(model, targets: set[str], *, max_layers: Optional[int]) -> list[tuple[str, nn.Linear]]:
    """Every targeted Linear (by leaf name), optionally limited to the first `max_layers` layers."""
    mods = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    if max_layers is not None:
        mods = [(n, m) for n, m in mods if _in_first_layers(n, max_layers)]
    return mods


# ---------------------------------------------------------------------------------------------
# Preprocessing (SVD) + basis storage
# ---------------------------------------------------------------------------------------------
def measure_preprocessing(model, cfg: dict[str, Any], device, *, max_layers: Optional[int]) -> dict:
    """Time the FP32 SVD that produces U_k, V_k for every targeted module; sum basis bytes.

    Mirrors adapters.LoRAOPLoRALinear exactly: w0 = W0.float(); U,_,Vh = svd(full_matrices=False);
    U_k = U[:, :k]; V_k = Vh[:k, :]^T — computed in basis_dtype (FP32). Each module is timed with a
    CUDA sync on both sides so the GPU SVD is actually captured.
    """
    adapter_cfg = cfg["adapter"]
    targets = set(adapter_cfg["targets"])
    protect_k = int(adapter_cfg["protect_k"])
    basis_dtype = getattr(torch, adapter_cfg.get("basis_dtype", "float32"))
    bytes_per = torch.empty(0, dtype=basis_dtype).element_size()

    mods = _targeted_linears(model, targets, max_layers=max_layers)
    per_module: list[dict[str, Any]] = []
    total_seconds = 0.0
    total_bytes = 0

    for name, module in mods:
        _sync(device)
        t0 = time.perf_counter()
        w0 = module.weight.detach().to(torch.float32)            # (d_out, d_in)
        U, _S, Vh = torch.linalg.svd(w0, full_matrices=False)
        k = min(protect_k, U.shape[1])
        U_k = U[:, :k].contiguous().to(basis_dtype)              # (d_out, k)
        V_k = Vh[:k, :].transpose(0, 1).contiguous().to(basis_dtype)  # (d_in, k)
        _sync(device)
        dt = time.perf_counter() - t0

        basis_bytes = (U_k.numel() + V_k.numel()) * bytes_per
        total_seconds += dt
        total_bytes += basis_bytes
        per_module.append({
            "name": name,
            "shape": [int(module.out_features), int(module.in_features)],
            "k": int(k),
            "svd_seconds": dt,
            "basis_bytes": int(basis_bytes),
        })

    n = len(mods)
    return {
        "protect_k": protect_k,
        "basis_dtype": adapter_cfg.get("basis_dtype", "float32"),
        "n_modules": n,
        "max_layers": max_layers,
        "total_svd_seconds": total_seconds,
        "mean_svd_seconds_per_module": (total_seconds / n) if n else 0.0,
        "basis_storage_bytes": int(total_bytes),
        "basis_storage_mib": total_bytes / (1024 ** 2),
        "per_module": per_module,
    }


# ---------------------------------------------------------------------------------------------
# Export overhead: projection + merge, OPLoRA vs LoRA-16
# ---------------------------------------------------------------------------------------------
def _build_adapter_model(config_path: str, device):
    """Load an adapter run's FP16 backbone + attach its adapters; load trained factors if present.

    Returns (model, adapters, loaded_trained). `loaded_trained` is False if the final checkpoint is
    absent (then the freshly-initialized factors are used — the export timing is shape-determined,
    so it stays representative).
    """
    cfg = load_yaml(config_path)
    cfg.setdefault("run_id", run_id_from_config(config_path, cfg))
    model, _tokenizer = load_base_model(cfg)
    model, adapters, _ = configure_method(model, cfg)
    model.to(device)

    ckpt = checkpoint_path(str(task1_dir("checkpoints") / cfg["run_id"]), _FINAL_UPDATE)
    loaded = False
    try:
        _load_trained_weights(model, cfg["method"], ckpt)
        loaded = True
    except (FileNotFoundError, OSError):
        log.warning("No final checkpoint at %s; timing export on initialized factors "
                    "(shape-determined, still representative).", ckpt)
    return model, adapters, loaded


def _time_projection(adapters: dict, device) -> float:
    """Time computing the effective (projected for OPLoRA, identity for LoRA) factors for all modules."""
    _sync(device)
    t0 = time.perf_counter()
    for adapter in adapters.values():
        a_eff, b_eff = adapter.effective_factors()
        # Touch the results so lazy CUDA kernels actually run before the sync.
        a_eff.sum(); b_eff.sum()
    _sync(device)
    return time.perf_counter() - t0


def _time_merge(model, adapters: dict, device) -> float:
    """Time merging the adapters into a fresh backbone (deep-copy + dW add) — shared by both methods."""
    _sync(device)
    t0 = time.perf_counter()
    merge_into_backbone(model, adapters)
    _sync(device)
    return time.perf_counter() - t0


def _subset_adapters(adapters: dict, max_layers: Optional[int]) -> dict:
    """Restrict an adapters handle to the first `max_layers` layers (for --smoke)."""
    if max_layers is None:
        return adapters
    return {n: a for n, a in adapters.items() if _in_first_layers(n, max_layers)}


def measure_export(oplora_config: str, lora_config: str, device, *, max_layers: Optional[int]) -> dict:
    """Projection + merge timing for OPLoRA vs LoRA-16; the delta isolates OPLoRA's projection."""
    results: dict[str, Any] = {}
    for key, cfg_path in (("oplora", oplora_config), ("lora_r16", lora_config)):
        model, adapters, loaded = _build_adapter_model(cfg_path, device)
        sub = _subset_adapters(adapters, max_layers)
        proj_s = _time_projection(sub, device)
        merge_s = _time_merge(model, sub, device)
        results[key] = {
            "n_modules": len(sub),
            "loaded_trained": loaded,
            "projection_seconds": proj_s,
            "merge_seconds": merge_s,
            "total_seconds": proj_s + merge_s,
        }
        del model, adapters
        if device.type == "cuda":
            torch.cuda.empty_cache()

    op, lo = results["oplora"], results["lora_r16"]
    results["projection_overhead_seconds"] = op["projection_seconds"] - lo["projection_seconds"]
    results["total_overhead_seconds"] = op["total_seconds"] - lo["total_seconds"]
    return results


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------
def run(oplora_config: str, lora_config: str, device, *, smoke: bool = False) -> dict:
    """Measure preprocessing + basis storage + export overhead; write oplora_overhead.json."""
    max_layers = _SMOKE_LAYERS if smoke else None

    cfg = load_yaml(oplora_config)
    cfg.setdefault("run_id", run_id_from_config(oplora_config, cfg))
    model, _tok = load_base_model(cfg)
    model.to(device)

    print(f"[oplora-overhead] preprocessing SVD on {device}"
          f"{f' (first {max_layers} layers)' if max_layers else ''} ...", flush=True)
    preprocessing = measure_preprocessing(model, cfg, device, max_layers=max_layers)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[oplora-overhead] export overhead (projection + merge; OPLoRA vs LoRA-16) ...", flush=True)
    export = measure_export(oplora_config, lora_config, device, max_layers=max_layers)

    fp32_master = bool(cfg.get("precision", {}).get("fp32_master", False))
    out = {
        "device": str(device),
        "backbone_dtype": "float32" if fp32_master else cfg.get("precision", {}).get("compute_dtype", "float16"),
        "smoke": smoke,
        "oplora_config": oplora_config,
        "lora_config": lora_config,
        "preprocessing": preprocessing,
        "basis_storage": {
            "total_bytes": preprocessing["basis_storage_bytes"],
            "total_mib": preprocessing["basis_storage_mib"],
            "dtype": preprocessing["basis_dtype"],
        },
        "export": export,
    }

    out_dir = ensure_dir(RESULTS_DIR / "task1")
    out_path = out_dir / "oplora_overhead.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[oplora-overhead] basis storage {preprocessing['basis_storage_mib']:.2f} MiB over "
          f"{preprocessing['n_modules']} modules; SVD {preprocessing['total_svd_seconds']:.3f}s; "
          f"projection overhead {export['projection_overhead_seconds']*1e3:.2f} ms; "
          f"wrote {out_path}.", flush=True)
    return out


def main() -> None:
    """CLI: --config (OPLoRA run), --lora-config (LoRA-16 baseline), --smoke, --drive-root."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser("OPLoRA overhead: basis storage, preprocessing time, export overhead.")
    parser.add_argument("--lora-config", default="configs/task1/lora_r16.yaml",
                        help="LoRA-16 config to compare the merge against (no projection).")
    parser.add_argument("--smoke", action="store_true", help="Restrict to the first 2 layers.")
    args = parser.parse_args()
    if args.drive_root:
        import os
        os.environ["DRIVE_ROOT"] = args.drive_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args.config, args.lora_config, device, smoke=args.smoke)


if __name__ == "__main__":
    main()
