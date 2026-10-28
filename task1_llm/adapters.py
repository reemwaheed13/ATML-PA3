"""Shared LoRA / OPLoRA adapter module with a `protect_k` option. (handout p4, Steps 3-4)

Design decision (approved): a single custom module serves both LoRA and OPLoRA.
  * protect_k == 0  -> plain LoRA:   W = W0 + (alpha/r) B A
  * protect_k >  0  -> OPLoRA:       orthogonally project the correction off the leading singular
                                     subspaces of W0 before applying it.

Why custom (not PEFT): OPLoRA must reuse LoRA-16's *initial unprojected* A/B factors bit-for-bit
(Step 4). Owning the random draw makes that trivial (same seeded generator, same order); with
PEFT we'd have to reproduce its internal init RNG and still subclass its Linear for the projected
forward. See README "Attribution" and the design note.

OPLoRA math (Step 4): for a starting weight W0 = U S V^T, with leading bases U_k, V_k (rank k):
    B_tilde = B - U_k (U_k^T B)
    A_tilde = A - (A V_k) V_k^T
    dW      = (alpha / r) B_tilde A_tilde
Bases computed once in FP32; use EFFECTIVE factors (not dense d_out x d_out / d_in x d_in
projection matrices); export the projected factors for inference.

Precision: adapters are FP32 parameters over a frozen FP16 backbone (p3). The random A draw is
always done on CPU from the supplied generator and then moved to the backbone device, so the
init is bit-identical regardless of whether the run is on CPU or GPU (this is what guarantees
LoRA-16 and OPLoRA-16 share factors even across devices).
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterConfig:
    rank: int
    alpha: int
    dropout: float
    protect_k: int = 0            # 0 => plain LoRA
    init_a: str = "random"
    init_b: str = "zeros"
    trainable_bias: bool = False
    basis_dtype: str = "float32"


def adapter_config_from_dict(adapter_cfg: dict[str, Any]) -> AdapterConfig:
    """Build an AdapterConfig from a run config's `adapter:` block (ignores `targets`/`kind`)."""
    return AdapterConfig(
        rank=int(adapter_cfg["rank"]),
        alpha=int(adapter_cfg["alpha"]),
        dropout=float(adapter_cfg.get("dropout", 0.0)),
        protect_k=int(adapter_cfg.get("protect_k", 0)),
        init_a=adapter_cfg.get("init_a", "random"),
        init_b=adapter_cfg.get("init_b", "zeros"),
        trainable_bias=bool(adapter_cfg.get("trainable_bias", False)),
        basis_dtype=adapter_cfg.get("basis_dtype", "float32"),
    )


class LoRAOPLoRALinear(nn.Module):
    """Wraps a frozen base nn.Linear with an FP32 low-rank correction (LoRA or OPLoRA).

    Wraps a target Linear (q/k/v/o/gate/up/down_proj). Holds:
      * frozen base weight W0 (FP16),
      * FP32 factors A (r x d_in), B (d_out x r),
      * for OPLoRA: cached leading bases U_k (d_out x k), V_k (d_in x k) computed once in FP32.
    """

    def __init__(self, base_linear: nn.Linear, cfg: AdapterConfig, *, generator=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)                       # frozen FP16 backbone
        if cfg.trainable_bias and self.base.bias is not None:
            self.base.bias.requires_grad_(True)           # (handout keeps this off)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = cfg.rank
        self.scaling = cfg.alpha / cfg.rank
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        device = base_linear.weight.device

        # --- factors -------------------------------------------------------------------------
        # A: drawn on CPU from `generator` so the init is device-independent and reproducible.
        if cfg.init_a == "random":
            # Kaiming-uniform(a=sqrt(5)) reduces to U(-1/sqrt(d_in), 1/sqrt(d_in)) for a (r,d_in)
            # weight — the standard LoRA A init, done explicitly so we control the RNG stream.
            bound = 1.0 / math.sqrt(self.in_features)
            a_cpu = torch.empty(cfg.rank, self.in_features, dtype=torch.float32)
            if generator is not None:
                a_cpu.uniform_(-bound, bound, generator=generator)
            else:
                a_cpu.uniform_(-bound, bound)
        elif cfg.init_a == "zeros":
            a_cpu = torch.zeros(cfg.rank, self.in_features, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown init_a {cfg.init_a!r}")

        if cfg.init_b not in ("zeros",):
            raise ValueError(f"init_b must be 'zeros' (LoRA), got {cfg.init_b!r}")

        self.A = nn.Parameter(a_cpu.to(device))
        self.B = nn.Parameter(torch.zeros(self.out_features, cfg.rank, dtype=torch.float32,
                                          device=device))

        # --- OPLoRA bases (computed once, in FP32) -------------------------------------------
        self.protect_k = int(cfg.protect_k)
        if self.protect_k > 0:
            basis_dtype = getattr(torch, cfg.basis_dtype)
            w0 = self.base.weight.detach().to(torch.float32)          # (d_out, d_in)
            # full_matrices=False -> U (d_out, m), Vh (m, d_in), m = min(d_out, d_in)
            U, _S, Vh = torch.linalg.svd(w0, full_matrices=False)
            k = min(self.protect_k, U.shape[1])
            if k < self.protect_k:
                # Defensive: protection rank cannot exceed the matrix rank bound.
                self.protect_k = k
            U_k = U[:, :k].contiguous().to(basis_dtype)                # (d_out, k)
            V_k = Vh[:k, :].transpose(0, 1).contiguous().to(basis_dtype)  # (d_in, k)
            self.register_buffer("U_k", U_k.to(device), persistent=True)
            self.register_buffer("V_k", V_k.to(device), persistent=True)
        else:
            self.U_k = None
            self.V_k = None

    # -- effective (possibly projected) factors ----------------------------------------------

    def _project(self):
        """Return (A_eff, B_eff). For OPLoRA, two-sidedly project the correction off U_k / V_k.

        Uses only k-wide intermediates (A@V_k, U_k^T@B) — never the dense d x d projectors.
        """
        if self.protect_k > 0:
            A_eff = self.A - (self.A @ self.V_k) @ self.V_k.transpose(0, 1)
            B_eff = self.B - self.U_k @ (self.U_k.transpose(0, 1) @ self.B)
            return A_eff, B_eff
        return self.A, self.B

    def forward(self, x):
        """Return base(x) + scaled low-rank correction (projected for OPLoRA)."""
        base_out = self.base(x)
        A_eff, B_eff = self._project()
        h = self.dropout(x)
        low = F.linear(h, A_eff)        # (..., r)
        low = F.linear(low, B_eff)      # (..., d_out)
        return base_out + self.scaling * low

    def effective_factors(self):
        """Return the factors used for inference/merge. LoRA: (A, B). OPLoRA: projected (A~, B~)."""
        return self._project()

    def delta_weight(self) -> torch.Tensor:
        """Return dW = (alpha/r) * B_eff @ A_eff (d_out x d_in), in FP32. For merge / diagnostics."""
        A_eff, B_eff = self.effective_factors()
        return self.scaling * (B_eff.to(torch.float32) @ A_eff.to(torch.float32))


def attach_adapters(
    model,
    cfg: AdapterConfig,
    target_names: list[str],
    *,
    generator=None,
) -> dict:
    """Replace each targeted Linear in every transformer layer with a LoRAOPLoRALinear.

    Returns a handle mapping module path -> adapter (ordered by model.named_modules()), so
    matched-init and export can be verified. The adapters are constructed in that deterministic
    order from the single `generator`, so two runs with the same targets/model draw bit-identical
    A factors (this is OPLoRA-16 reusing LoRA-16's init). `generator` should be a CPU generator.
    """
    targets = set(target_names)
    module_lookup = dict(model.named_modules())

    to_replace = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]

    handle: dict[str, LoRAOPLoRALinear] = {}
    for name, module in to_replace:
        parent_name, _, attr = name.rpartition(".")
        parent = module_lookup[parent_name] if parent_name else model
        adapter = LoRAOPLoRALinear(module, cfg, generator=generator)
        setattr(parent, attr, adapter)
        handle[name] = adapter
    return handle


def export_initial_factors(adapters: dict, path: str) -> None:
    """Persist the initial UNPROJECTED A/B factors so OPLoRA can match LoRA-16's init (Step 4).

    Call this immediately after attach_adapters and BEFORE any optimizer step, so the saved
    factors are the pristine initialization.
    """
    path = os.fspath(path)
    state = {
        name: {"A": adapter.A.detach().cpu().clone(), "B": adapter.B.detach().cpu().clone()}
        for name, adapter in adapters.items()
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def _resolve_initial_factors_path(source: str, factors_dir: Optional[str]) -> str:
    """Resolve where a run's exported initial factors live.

    If `source` is an existing file, use it directly. Otherwise treat it as a run_id and build
    <factors_dir>/<run_id>_initial_factors.pt, defaulting factors_dir to DRIVE_ROOT/task1/
    checkpoints (resolved lazily via common.paths so importing this module needs no DRIVE_ROOT).
    """
    if os.path.exists(str(source)):
        return os.fspath(source)
    if factors_dir is None:
        from common.paths import task1_dir
        factors_dir = os.fspath(task1_dir("checkpoints"))
    return os.path.join(factors_dir, f"{source}_initial_factors.pt")


def load_matched_init(
    adapters: dict,
    source_run_id: str,
    *,
    factors_dir: Optional[str] = None,
    map_location: str = "cpu",
) -> None:
    """Load initial unprojected A/B factors from another run (OPLoRA reuses LoRA-16's). (Step 4)

    Copies the saved factors into `adapters` in place, asserting the module paths and shapes
    match. Must be called before any optimizer step so the init is bit-identical to the source.
    """
    path = _resolve_initial_factors_path(source_run_id, factors_dir)
    try:
        saved = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location=map_location)

    missing = set(adapters) - set(saved)
    if missing:
        raise KeyError(f"Matched-init source {source_run_id!r} is missing factors for: {sorted(missing)}")

    with torch.no_grad():
        for name, adapter in adapters.items():
            src = saved[name]
            if src["A"].shape != adapter.A.shape or src["B"].shape != adapter.B.shape:
                raise ValueError(
                    f"Shape mismatch at {name}: saved A{tuple(src['A'].shape)} "
                    f"B{tuple(src['B'].shape)} vs adapter A{tuple(adapter.A.shape)} "
                    f"B{tuple(adapter.B.shape)}"
                )
            adapter.A.copy_(src["A"].to(adapter.A.device, adapter.A.dtype))
            adapter.B.copy_(src["B"].to(adapter.B.device, adapter.B.dtype))


def merge_into_backbone(model, adapters: dict) -> Any:
    """Return a fresh deep-copied backbone with adapters merged (projected factors for OPLoRA).

    Produces a model with plain Linears (adapter wrappers replaced by their base, dW added into
    the weight) so evaluation runs the merged weights with no adapter path. The caller verifies
    merged vs unmerged logits agree on a small batch (Step 5). dW is added in the backbone
    weight's dtype (FP16), matching how eval runs.
    """
    merged = copy.deepcopy(model)
    merged_lookup = dict(merged.named_modules())

    for name, adapter in adapters.items():
        merged_adapter = merged_lookup[name]        # the deep-copied LoRAOPLoRALinear
        base = merged_adapter.base
        dW = adapter.delta_weight().to(base.weight.dtype).to(base.weight.device)
        with torch.no_grad():
            base.weight.add_(dW)

        parent_name, _, attr = name.rpartition(".")
        parent = merged_lookup[parent_name] if parent_name else merged
        setattr(parent, attr, base)                 # drop the wrapper, keep the merged Linear

    return merged
