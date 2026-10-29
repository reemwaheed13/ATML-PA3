"""Assemble the VLM: insert visual tokens into Qwen's chat, with LoRA on the LM. (handout p9 Step 1)

Builds the forward path cached-features -> connector -> visual tokens inserted into the user
message BEFORE the question, preserving Qwen's chat structure. Constructs matching labels,
attention masks, and position indices; masks visual tokens / prompts / padding in the loss. The
Qwen backbone is frozen (FP16); LoRA adapters (reuse task1_llm.adapters) are attached from the
start (B initialized to 0 => zero contribution until Stage 2 adds them to the optimizer), targeting
q/k/v/o_proj. Autograd must still reach the connector even though the backbone is frozen ("What to
Watch For", p9): we pass `inputs_embeds` that carry the connector's grad history and never wrap the
forward in no_grad.

For the no-image condition, visual tokens are omitted and masks/positions rebuilt (p8).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.precision import PrecisionConfig, autocast_context
from task1_llm.adapters import adapter_config_from_dict, attach_adapters
from .connector import Connector, pool_2x2

IGNORE_INDEX = -100


class SigLIPQwenVLM(nn.Module):
    """Frozen SigLIP features (precomputed) + trainable connector + frozen Qwen (+ LoRA)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        lm_cfg = cfg["models"]["language_model"]
        self.enable_thinking = bool(lm_cfg.get("enable_thinking", False))

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_cfg["name"])
        self.qwen = AutoModelForCausalLM.from_pretrained(lm_cfg["name"], dtype=torch.float16)
        self.qwen.config.use_cache = False
        if lm_cfg.get("gradient_checkpointing", True):
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        for p in self.qwen.parameters():
            p.requires_grad_(False)                       # frozen FP16 backbone
        # Let gradients reach the connector/adapters through the (frozen) embedding path under
        # gradient checkpointing.
        self.qwen.enable_input_require_grads()

        hidden = int(self.qwen.config.hidden_size)
        feat_dim = int(cfg["models"]["vision_encoder"]["feature_dim"])
        conn_cfg = cfg["models"]["connector"]
        self.connector = Connector(in_dim=feat_dim,
                                   hidden_dim=int(conn_cfg.get("hidden_dim", 1024)),
                                   out_dim=int(conn_cfg.get("out_dim", hidden)))

        # Optional 2x2 pooling to 49 visual tokens (compress49 experiment, p9 Step 7).
        self.num_visual_tokens = int(cfg.get("visual_tokens", {}).get("count", 196))
        self.pool_visual = self.num_visual_tokens == 49

        # LoRA on the LM (attached always; zero-contribution until Stage 2 optimizes it).
        self.adapters: dict[str, Any] = {}
        if "adapter" in cfg:
            acfg = adapter_config_from_dict(cfg["adapter"])
            from common import SEED
            generator = torch.Generator().manual_seed(SEED)
            self.adapters = attach_adapters(self.qwen, acfg, list(cfg["adapter"]["targets"]),
                                            generator=generator)
        # Adapters start FROZEN: Stage 1 is connector-only. train.py unfreezes them at Stage 2
        # (B is zero-initialized, so a frozen adapter contributes nothing to the Stage-1 forward).
        self.set_adapter_trainable(False)

    # -- devices / params -----------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def embed_tokens(self):
        return self.qwen.get_input_embeddings()

    def connector_parameters(self):
        return list(self.connector.parameters())

    def adapter_parameters(self):
        params = []
        for ad in self.adapters.values():
            params += [ad.A, ad.B]
        return params

    def set_adapter_trainable(self, flag: bool) -> None:
        """Freeze/unfreeze every adapter A/B factor (Stage 1 frozen; Stage 2+ trainable)."""
        for ad in self.adapters.values():
            ad.A.requires_grad_(flag)
            ad.B.requires_grad_(flag)

    def trainable_param_counts(self) -> dict[str, int]:
        """Trainable parameter counts by group, derived from requires_grad (for per-stage logging)."""
        conn = sum(p.numel() for p in self.connector_parameters() if p.requires_grad)
        adapt = sum(p.numel() for p in self.adapter_parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"connector": int(conn), "adapters": int(adapt), "total": int(total)}

    # -- input assembly -------------------------------------------------------------------------

    def _visual_embeds(self, features: torch.Tensor, pcfg: PrecisionConfig,
                       amp_enabled: bool) -> torch.Tensor:
        """Connector outputs [B, n_visual, H] from cached features [B, 196, 768] (autocast)."""
        feats = features.to(self.device)
        if self.pool_visual:
            feats = pool_2x2(feats)
        with autocast_context(pcfg, device_type=self.device.type, enabled=amp_enabled):
            return self.connector(feats)

    def build_inputs(self, examples, visual: Optional[torch.Tensor], *, with_image: bool = True,
                     pad_side: str = "right"):
        """Splice visual embeddings into each example and pad to a batch.

        Returns (inputs_embeds [B,T,H], attention_mask [B,T], position_ids [B,T], labels [B,T]).
        For with_image the 196 (or 49) visual embeddings are inserted at each example's `insert_at`
        with IGNORE labels; for no-image the text sequence is used as-is (masks/positions rebuilt).
        """
        embed = self.embed_tokens()
        device = self.device
        rows_embeds, rows_labels = [], []

        for i, ex in enumerate(examples):
            ids = torch.tensor(ex.input_ids, dtype=torch.long, device=device)
            txt = embed(ids)                                  # [L, H] FP16, frozen
            labels = list(ex.labels)
            if with_image and visual is not None:
                v = visual[i].to(txt.dtype)                   # [n_visual, H]
                at = ex.insert_at
                emb = torch.cat([txt[:at], v, txt[at:]], dim=0)
                lab = labels[:at] + [IGNORE_INDEX] * v.shape[0] + labels[at:]
            else:
                emb = txt
                lab = labels
            rows_embeds.append(emb)
            rows_labels.append(torch.tensor(lab, dtype=torch.long, device=device))

        return self._pad_batch(rows_embeds, rows_labels, pad_side=pad_side)

    def _pad_batch(self, rows_embeds, rows_labels, *, pad_side: str):
        """Pad variable-length rows into batch tensors; position_ids via the attention mask."""
        device = self.device
        H = rows_embeds[0].shape[-1]
        dtype = rows_embeds[0].dtype
        T = max(e.shape[0] for e in rows_embeds)
        B = len(rows_embeds)

        embeds = torch.zeros(B, T, H, dtype=dtype, device=device)
        attn = torch.zeros(B, T, dtype=torch.long, device=device)
        labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long, device=device)
        for i, (e, lab) in enumerate(zip(rows_embeds, rows_labels)):
            n = e.shape[0]
            sl = slice(0, n) if pad_side == "right" else slice(T - n, T)
            embeds[i, sl] = e
            attn[i, sl] = 1
            labels[i, sl] = lab
        # Position ids independent of pad side: first valid token -> 0.
        position_ids = attn.cumsum(-1).sub(1).clamp_min(0)
        return embeds, attn, position_ids, labels

    # -- training -------------------------------------------------------------------------------

    def forward(self, examples, features, *, pcfg: PrecisionConfig, amp_enabled: bool,
                with_image: bool = True):
        """Example-weighted causal CE over one microbatch; returns {"loss", "n_examples"}.

        Loss (p8): average supervised-token NLL within each example, then across the batch. Each
        batch row is exactly one example, so a per-row token-mean followed by a row-mean implements
        the example weighting directly.
        """
        visual = self._visual_embeds(features, pcfg, amp_enabled) if with_image else None
        embeds, attn, pos, labels = self.build_inputs(examples, visual, with_image=with_image)

        with autocast_context(pcfg, device_type=self.device.type, enabled=amp_enabled):
            out = self.qwen(inputs_embeds=embeds, attention_mask=attn, position_ids=pos)
        logits = out.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        B, Tm1, V = shift_logits.shape
        tok = F.cross_entropy(shift_logits.reshape(-1, V).float(), shift_labels.reshape(-1),
                              ignore_index=IGNORE_INDEX, reduction="none").view(B, Tm1)
        mask = (shift_labels != IGNORE_INDEX).float()
        counts = mask.sum(1)
        per_example = (tok * mask).sum(1) / counts.clamp_min(1.0)
        valid = counts > 0
        loss = per_example[valid].mean() if valid.any() else per_example.sum() * 0.0
        return {"loss": loss, "n_examples": int(valid.sum())}

    # -- generation -----------------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, examples, features, *, max_new_tokens: int, pcfg: PrecisionConfig,
                 amp_enabled: bool, with_image: bool = True, greedy: bool = True):
        """Greedy generation under an image condition; returns a list of decoded answer strings.

        Examples carry only the prompt (empty answer => labels all IGNORE). Left-padding is used so
        batched decoder generation aligns the newest token across rows.
        """
        visual = self._visual_embeds(features, pcfg, amp_enabled) if with_image else None
        embeds, attn, pos, _ = self.build_inputs(examples, visual, with_image=with_image,
                                                 pad_side="left")
        with autocast_context(pcfg, device_type=self.device.type, enabled=amp_enabled):
            gen = self.qwen.generate(
                inputs_embeds=embeds, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=not greedy, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        # With inputs_embeds, generate returns only the newly generated token ids.
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen]


def sanity_checks(model: SigLIPQwenVLM, cfg: dict[str, Any], *, n: int = 2) -> dict[str, Any]:
    """Verify frozen parameter sets, connector gradients, feature/visual shapes, split isolation. (p9)

    Runs ONE real forward/backward on cached features for `n` manifest images and asserts that every
    connector parameter has a non-None, finite, nonzero gradient (autograd actually reaches the
    connector through the frozen backbone). Also asserts feature shape [B,196,768] and visual-embed
    shape [B,196,1024] (or [B,49,768]->[B,49,1024] for compress49). Raises on any failure.
    """
    import json
    import os
    from common.paths import REPO_ROOT
    from .data import VLMExample, format_vlm_example
    from .features import load_cached_features

    report: dict[str, Any] = {}

    # --- frozen parameter sets ---------------------------------------------------------------
    backbone_trainable = [nm for nm, p in model.qwen.named_parameters()
                          if p.requires_grad and not _is_adapter_param(model, p)]
    report["qwen_backbone_all_frozen"] = len(backbone_trainable) == 0
    report["connector_trainable"] = all(p.requires_grad for p in model.connector_parameters())
    report["num_adapters"] = len(model.adapters)
    assert report["qwen_backbone_all_frozen"], f"Qwen backbone not fully frozen: {backbone_trainable[:5]}"
    assert report["connector_trainable"], "connector parameters are not all trainable"

    # --- load a small real batch of cached features ------------------------------------------
    manifest_path = os.fspath(REPO_ROOT / cfg.get("manifest", "splits/task2/image_manifest.json"))
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    batch: list[tuple[int, torch.Tensor]] = []
    for image_id in manifest:
        try:
            feats = load_cached_features(cfg, int(image_id), device=model.device)
        except Exception:
            continue
        batch.append((int(image_id), feats))
        if len(batch) >= n:
            break
    if not batch:
        raise RuntimeError("sanity_checks needs cached features; run task2_vlm.features first.")

    ve = cfg["models"]["vision_encoder"]
    num_patches, feat_dim = int(ve.get("num_patches", 196)), int(ve["feature_dim"])
    features = torch.stack([f for _, f in batch])
    assert tuple(features.shape[1:]) == (num_patches, feat_dim), \
        f"feature shape {tuple(features.shape)} != (*, {num_patches}, {feat_dim})"
    report["feature_shape"] = list(features.shape)

    # --- visual-embed shapes (incl. compress49 pooled path) ----------------------------------
    pcfg = PrecisionConfig(compute_dtype=cfg.get("precision", {}).get("compute_dtype", "float16"),
                           fp32_master=False, grad_clip_norm=1.0)
    amp = model.device.type == "cuda"
    out_dim = int(cfg["models"]["connector"].get("out_dim", model.qwen.config.hidden_size))
    exp_visual = 49 if model.pool_visual else num_patches
    if model.pool_visual:
        assert tuple(pool_2x2(features).shape[1:]) == (49, feat_dim), \
            f"pooled feature shape {tuple(pool_2x2(features).shape)} != (*, 49, {feat_dim})"
    visual = model._visual_embeds(features, pcfg, amp)
    assert tuple(visual.shape) == (len(batch), exp_visual, out_dim), \
        f"visual shape {tuple(visual.shape)} != ({len(batch)}, {exp_visual}, {out_dim})"
    report["visual_shape"] = list(visual.shape)
    report["num_visual_tokens"] = exp_visual

    # --- one forward/backward: connector grads must be non-None, finite, nonzero --------------
    caps = cfg.get("train", {}).get("content_token_caps", {})
    examples = []
    for image_id, _ in batch:
        fe = format_vlm_example(model.tokenizer, "Describe this image.", "a test caption",
                                answer_cap=int(caps.get("caption_target", 64)),
                                question_cap=int(caps.get("question", 80)),
                                enable_thinking=model.enable_thinking)
        examples.append(VLMExample(image_id=image_id, component="sanity", input_ids=fe["input_ids"],
                                   labels=fe["labels"], insert_at=fe["insert_at"],
                                   n_answer_tokens=fe["n_answer_tokens"]))

    model.zero_grad(set_to_none=True)
    out = model(examples, features, pcfg=pcfg, amp_enabled=amp, with_image=True)
    out["loss"].backward()
    for name, p in model.connector.named_parameters():
        assert p.grad is not None, f"connector.{name} grad is None (autograd not reaching connector)"
        assert torch.isfinite(p.grad).all(), f"connector.{name} grad has non-finite values"
        assert float(p.grad.abs().sum()) > 0.0, f"connector.{name} grad is all zeros"
    model.zero_grad(set_to_none=True)
    report["connector_grad_ok"] = True

    # --- split isolation ---------------------------------------------------------------------
    report["manifest_images"] = len(manifest)
    report["subsets_present"] = sorted(set(manifest.values()))   # one subset per image id by design
    return report


def _is_adapter_param(model: SigLIPQwenVLM, p) -> bool:
    for ad in model.adapters.values():
        if p is ad.A or p is ad.B:
            return True
    return False
