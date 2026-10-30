"""Activation diagnostics: local update response R(z) on math vs non-math prompts.
(handout p5, Step 6)

Probe set (p3 "Diagnostic and qualitative inputs"): 32 GSM8K test questions + 32 HellaSwag
validation contexts, sampled with seed 6304 and frozen to splits/task1/activation_probe_ids.json
(path from configs/task1/diagnostics.yaml). Sampling depends only on dataset sizes + the seed,
never on any model output, so the probe set is fixed before any model is touched and reused for
every checkpoint.

Prompt formats match evaluation:
  * GSM8K    = the gsm8k_cot zero-shot NON-thinking chat prompt (apply_chat_template,
               enable_thinking=False, generation prompt appended) — the same prompt path
               evaluate.py uses for gsm8k_cot. z is taken at the last prompt token.
  * HellaSwag= the stock lm-eval context (activity_label + ctx_a + ctx_b.capitalize(),
               preprocessed) with NO ending appended and NO chat template. z is taken at the
               last context token.

On the ORIGINAL model only, forward pre-hooks on down_proj of layers 0, 13, 27 cache the input z
at the last non-padding prompt token (one prompt at a time, so "last token" is position -1, no
padding). Then, for the six checkpoints and the SAME cached z and the SAME dW as the weight-space
diagnostics (task1_llm.diagnostics_common), compute per layer:

    R(z) = ||dW z||_2 / (||W0 z||_2 + 1e-8).

Reusing the cached z isolates the local weight change; it does not capture downstream changes
("What to Watch For", p5). Everything FP32. Per-example values + a mean/median summary (per
checkpoint x layer x math/non-math) go to results/task1/diagnostics_activation.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Optional

import numpy as np
import torch

from common.paths import REPO_ROOT, RESULTS_DIR, ensure_dir, task1_dir

from . import diagnostics_common as dc
from .data import GSM8K_CONFIG, GSM8K_DATASET

log = logging.getLogger(__name__)

# Defaults for the two probe sources; overridable under probe_set.<name> in the config.
_GSM8K_DEFAULTS = {"dataset": GSM8K_DATASET, "config": GSM8K_CONFIG, "split": "test",
                   "question_field": "question"}
_HELLASWAG_DEFAULTS = {"dataset": "Rowan/hellaswag", "config": None, "split": "validation"}


# ---------------------------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------------------------
def _hellaswag_preprocess(text: str) -> str:
    """lm-eval's hellaswag text cleanup (matches the harness so the context is identical)."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hellaswag_context(doc: dict[str, Any]) -> str:
    """Stock lm-eval HellaSwag context: activity_label + ctx_a + ctx_b.capitalize(), cleaned."""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return _hellaswag_preprocess(doc["activity_label"] + ": " + ctx)


def _flatten_ids(ids) -> list[int]:
    """Coerce a tokenizer input_ids result to a flat list[int] (unwrap an accidental batch dim)."""
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def gsm8k_prompt_text(tokenizer, question: str, template: str, enable_thinking: bool) -> str:
    """Render the gsm8k_cot non-thinking chat prompt STRING (user content from `template`).

    Renders with tokenize=False (stable across transformers versions; see data._render_token_ids)
    and a generation prompt, enable_thinking=False — exactly the prompt evaluate.py fed the model
    for gsm8k_cot, so it can be compared byte-for-byte against the stored lm-eval context.
    """
    content = template.format(question=question)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    if not isinstance(text, str):
        text = text[0] if isinstance(text, (list, tuple)) and text else str(text)
    return text


def _gsm8k_prompt_ids(tokenizer, text: str) -> list[int]:
    """Token ids for an already-rendered chat prompt string (template placed the special tokens)."""
    return _flatten_ids(tokenizer(text, add_special_tokens=False)["input_ids"])


def _hellaswag_prompt_ids(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    """Stock completion-prompt ids for a HellaSwag context (no chat template)."""
    return _flatten_ids(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


# ---------------------------------------------------------------------------------------------
# Probe-set construction (seed 6304, model-independent)
# ---------------------------------------------------------------------------------------------
def _ids_path(cfg: dict[str, Any]) -> str:
    rel = cfg["probe_set"].get("ids_file", "splits/task1/activation_probe_ids.json")
    return os.fspath(rel if os.path.isabs(rel) else REPO_ROOT / rel)


def _source_coords(cfg: dict[str, Any], name: str, defaults: dict) -> dict:
    coords = dict(defaults)
    coords.update(cfg["probe_set"].get(name, {}) or {})
    return coords


def _load_source(coords: dict):
    from datasets import load_dataset

    if coords.get("config"):
        return load_dataset(coords["dataset"], coords["config"], split=coords["split"])
    return load_dataset(coords["dataset"], split=coords["split"])


def create_probe_ids(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Sample and freeze the probe IDs (32 GSM8K + 32 HellaSwag, seed 6304) to the ids file.

    Sampling uses only the dataset lengths and the seed — never any model output — so the probe
    set is reproducible and fixed before evaluation. Refuses to overwrite unless force=True.
    """
    ps = cfg["probe_set"]
    out_path = _ids_path(cfg)
    if os.path.exists(out_path) and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    seed = int(ps.get("seed", 6304))
    gsm = _source_coords(cfg, "gsm8k", _GSM8K_DEFAULTS)
    hel = _source_coords(cfg, "hellaswag", _HELLASWAG_DEFAULTS)
    gsm_ds, hel_ds = _load_source(gsm), _load_source(hel)

    rng = np.random.default_rng(seed)
    gsm_idx = sorted(int(i) for i in rng.choice(len(gsm_ds), int(ps["n_gsm8k"]), replace=False))
    hel_idx = sorted(int(i) for i in rng.choice(len(hel_ds), int(ps["n_hellaswag"]), replace=False))

    payload = {
        "seed": seed,
        "gsm8k": {**gsm, "indices": gsm_idx},
        "hellaswag": {**hel, "indices": hel_idx},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[diag-act] wrote {len(gsm_idx)}+{len(hel_idx)} probe ids to {out_path}.", flush=True)
    return payload


def build_probes(cfg: dict[str, Any], tokenizer, probe_ids: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize probe prompts for the frozen GSM8K + HellaSwag IDs.

    Returns a list of {type: 'math'|'non_math', id, text, input_ids}. `text` is the exact prompt
    string the tokens come from (the chat-templated gsm8k_cot prompt, or the stock HellaSwag
    context with no ending), kept so the smoke check can compare it against lm-eval's prompt.
    """
    ps = cfg["probe_set"]
    enable_thinking = bool(cfg.get("model", {}).get("enable_thinking", False))
    gsm_template = ps.get("gsm8k_user_template", "{question}")
    hel_add_special = bool(ps.get("hellaswag_add_special_tokens", False))

    gsm, hel = probe_ids["gsm8k"], probe_ids["hellaswag"]
    gsm_ds, hel_ds = _load_source(gsm), _load_source(hel)
    qfield = gsm.get("question_field", "question")

    probes: list[dict[str, Any]] = []
    for i in gsm["indices"]:
        question = gsm_ds[int(i)][qfield]
        text = gsm8k_prompt_text(tokenizer, question, gsm_template, enable_thinking)
        probes.append({"type": "math", "id": int(i), "text": text,
                       "input_ids": _gsm8k_prompt_ids(tokenizer, text)})
    for i in hel["indices"]:
        text = hellaswag_context(hel_ds[int(i)])
        probes.append({"type": "non_math", "id": int(i), "text": text,
                       "input_ids": _hellaswag_prompt_ids(tokenizer, text, hel_add_special)})
    return probes


# ---------------------------------------------------------------------------------------------
# Smoke prompt check: compare our rendered GSM8K prompt against the one lm-eval actually used
# ---------------------------------------------------------------------------------------------
def _extract_lm_eval_context(arguments: Any) -> Optional[str]:
    """Pull the context (prompt) string out of a logged lm-eval sample's `arguments`.

    lm-eval 0.4.x stores the request args as the context followed by gen kwargs, as either a
    nested list ([[context, {...}]]) or a dict ({"gen_args_0": {"arg_0": context, ...}}). The
    context is always the first string encountered, so return that.
    """
    found: list[str] = []

    def walk(node: Any) -> None:
        if found:
            return
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
                if found:
                    return
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
                if found:
                    return

    walk(arguments)
    return found[0] if found else None


def _lookup_lm_eval_prompt(path: str, doc_id: int) -> Optional[str]:
    """Return the prompt lm-eval used for `doc_id` from a baseline per-example jsonl, or None."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if int(rec.get("doc_id", -1)) == int(doc_id):
                return _extract_lm_eval_context(rec.get("arguments"))
    return None


def smoke_prompt_check(probes: list[dict[str, Any]]) -> None:
    """Compare the first GSM8K probe's prompt to lm-eval's; print a HellaSwag context preview.

    The reference is the prompt lm-eval fed the baseline model, stored per-example at
    DRIVE_ROOT/task1/eval/baseline/gsm8k.jsonl (matched on doc_id). If the file is missing, warn
    instead of failing — the diagnostics proper do not depend on it.
    """
    math_probes = [p for p in probes if p["type"] == "math"]
    non_math = [p for p in probes if p["type"] == "non_math"]

    if math_probes:
        probe = math_probes[0]
        ours = probe["text"]
        path = os.fspath(task1_dir("eval") / "baseline" / "gsm8k.jsonl")
        print(f"\n[diag-act smoke] GSM8K prompt check (doc_id={probe['id']})", flush=True)
        if not os.path.exists(path):
            print(f"  WARNING: baseline samples file not found: {path}; skipping prompt match.")
        else:
            ref = _lookup_lm_eval_prompt(path, probe["id"])
            if ref is None:
                print(f"  WARNING: doc_id {probe['id']} not found in {path}; skipping prompt match.")
            else:
                print("  PROMPT MATCH" if ours == ref else "  PROMPT MISMATCH")
                print(f"  --- ours ---\n{ours}")
                print(f"  --- lm-eval ---\n{ref}")

    if non_math:
        print("\n[diag-act smoke] HellaSwag context (first 300 chars):", flush=True)
        print(non_math[0]["text"][:300])


# ---------------------------------------------------------------------------------------------
# Caching the original model's down_proj inputs z
# ---------------------------------------------------------------------------------------------
def cache_downproj_inputs(cfg: dict[str, Any], tokenizer, probes: list[dict[str, Any]],
                          layers: list[int], device) -> tuple[list[dict[int, torch.Tensor]],
                                                               dict[int, torch.Tensor]]:
    """Run the ORIGINAL model with forward pre-hooks; cache z at the last prompt token per probe.

    Returns (z_by_probe, w0_by_layer): z_by_probe[i][layer] is the FP32 CPU input vector to
    down_proj at the last token for probe i; w0_by_layer is the original down_proj weight.
    """
    from transformers import AutoModelForCausalLM

    model_name = dc.base_model_name(dc.resolve_train_cfg(cfg["checkpoints"][0]["run"]))
    log.info("Loading original model %s for activation caching.", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()

    modules = dc.down_proj_modules(model, layers)
    w0_by_layer = dc.extract_w0(model, layers)

    scratch: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, args):
            z = args[0]                                   # (batch, seq, d_in) input to down_proj
            scratch[layer] = z[:, -1, :].detach().to(torch.float32).cpu().squeeze(0).clone()
        return hook

    handles = [modules[layer].register_forward_pre_hook(make_hook(layer)) for layer in layers]
    z_by_probe: list[dict[int, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for probe in probes:
                scratch.clear()
                ids = torch.tensor([probe["input_ids"]], dtype=torch.long, device=device)
                model(input_ids=ids)
                z_by_probe.append({layer: scratch[layer] for layer in layers})
    finally:
        for h in handles:
            h.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return z_by_probe, w0_by_layer


def response_ratio(delta_w: torch.Tensor, w0: torch.Tensor, z: torch.Tensor, eps: float) -> float:
    """Return R(z) = ||dW z||_2 / (||W0 z||_2 + eps), all FP32."""
    dwz = float(torch.linalg.norm(delta_w @ z))
    w0z = float(torch.linalg.norm(w0 @ z))
    return dwz / (w0z + eps)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def _summarize(per_example: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/median of R per (run, update, layer, probe_type)."""
    groups: dict[tuple, list[float]] = {}
    meta: dict[tuple, dict] = {}
    for row in per_example:
        key = (row["run"], row["update"], row["layer"], row["probe_type"])
        groups.setdefault(key, []).append(row["R"])
        meta[key] = {"run": row["run"], "update": row["update"], "method": row["method"],
                     "layer": row["layer"], "probe_type": row["probe_type"]}
    summary = []
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        summary.append({**meta[key], "mean": float(arr.mean()),
                        "median": float(np.median(arr)), "n": int(arr.size)})
    return summary


def run(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Build/freeze probes, cache original-model z, compute R(z) for every checkpoint/layer/probe."""
    act = cfg["activation"]
    eps = float(act.get("epsilon", 1e-8))
    layers = list(cfg["weight_space"]["layers"])
    checkpoints = list(cfg["checkpoints"])

    probe_ids = create_probe_ids(cfg)                      # freeze (or load) the full 32+32 set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        dc.base_model_name(dc.resolve_train_cfg(checkpoints[0]["run"])))

    probes = build_probes(cfg, tokenizer, probe_ids)
    if smoke:
        checkpoints, layers = dc.apply_smoke(cfg, checkpoints, layers)
        sm = cfg.get("smoke", {}) or {}
        n_math = int(sm.get("n_gsm8k", 2))
        n_non = int(sm.get("n_hellaswag", 2))
        math_p = [p for p in probes if p["type"] == "math"][:n_math]
        non_p = [p for p in probes if p["type"] == "non_math"][:n_non]
        probes = math_p + non_p
        smoke_prompt_check(probes)  # verify our GSM8K prompt matches the one lm-eval evaluated

    z_by_probe, w0_by_layer = cache_downproj_inputs(cfg, tokenizer, probes, layers, device)

    per_example: list[dict[str, Any]] = []
    for spec in checkpoints:
        run_id, update = spec["run"], int(spec["update"])
        train_cfg = dc.resolve_train_cfg(run_id, spec.get("config"))
        method = train_cfg["method"]
        log.info("Activation diagnostics: %s @%d (%s).", run_id, update, method)
        weights = dc.load_checkpoint_weights(run_id, update)
        dW_by_layer = dc.build_delta_weights(train_cfg, weights, w0_by_layer, layers)

        for layer in layers:
            dW, w0 = dW_by_layer[layer], w0_by_layer[layer]
            for probe, z in zip(probes, z_by_probe):
                per_example.append({
                    "run": run_id, "update": update, "method": method, "layer": layer,
                    "probe_type": probe["type"], "example_id": probe["id"],
                    "R": response_ratio(dW, w0, z[layer], eps),
                })

    out = {
        "config": {"module": act.get("module", "down_proj"), "layers": layers, "epsilon": eps,
                   "compute_dtype": "float32", "smoke": smoke,
                   "ids_file": _ids_path(cfg), "n_probes": len(probes)},
        "per_example": per_example,
        "summary": _summarize(per_example),
    }
    out_dir = ensure_dir(RESULTS_DIR / "task1")
    out_path = out_dir / "diagnostics_activation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[diag-act] wrote {len(per_example)} per-example row(s) to {out_path}.", flush=True)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Task 1 activation diagnostics (Step 6).")
    parser.add_argument("--config", default="configs/task1/diagnostics.yaml",
                        help="Path to the diagnostics YAML config.")
    parser.add_argument("--drive-root", default=None, help="Override DRIVE_ROOT (else env var).")
    parser.add_argument("--make-probes", action="store_true",
                        help="(Re)generate the frozen probe-id file and exit (use --force to overwrite).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing probe-id file.")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast subset (one layer, few checkpoints, 2+2 probes) for Colab testing.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root

    from common.config import load_yaml
    cfg = load_yaml(args.config)
    if args.make_probes:
        create_probe_ids(cfg, force=args.force)
        return
    run(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
