"""Task 2 grounding evaluation: development + final. (handout p8-p9)

Development (before training, every 500 updates, stage endpoints): caption reference loss +
reference-answer NLL on both VQA dev suites under correct / complementary / no-image, and the
image-swap penalty delta_NLL = NLL(y|q, I_partner) - NLL(y|q, I_correct) (exclude termination;
average token NLL within example before averaging examples).

Final (both test suites, greedy, 16-new-token cap, batch 8) under four image conditions
(correct / complementary swap / mismatched / no image). Scores: grounding normalized exact match
(presence/counting separate); natural VQAv2 soft consensus accuracy (yes-no/number/other
separate). Also pair accuracy, identical-answer rate, empty-answer rate, mean reference-answer
NLL, and generation time per answer. Correct-image advantages over complementary swaps in accuracy
and NLL, with 95% CIs via 4,000 bootstrap resamples of complete pairs (matched pair IDs).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from common import SEED
from common.checkpoint import latest_checkpoint, load_checkpoint
from common.config import load_yaml
from common.paths import REPO_ROOT, ensure_dir, task2_dir
from .data import IGNORE_INDEX, format_vlm_example
from .features import load_cached_features
from .model import SigLIPQwenVLM
from .scoring import normalize_answer, normalized_exact_match, soft_consensus_accuracy

# -- caps / config access --------------------------------------------------------------------------

def _caps(cfg: dict[str, Any]) -> tuple[int, int, int]:
    c = cfg.get("train", {}).get("content_token_caps", {})
    return int(c.get("question", 80)), int(c.get("caption_target", 64)), int(c.get("short_answer_target", 16))


def _ids_path(cfg: dict[str, Any], name: str) -> str:
    return os.fspath(REPO_ROOT / cfg.get("ids_dir", "splits/task2") / name)


def _load(cfg: dict[str, Any], name: str) -> list:
    path = _ids_path(cfg, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -- eval item construction ------------------------------------------------------------------------

def _grounding_instances(pairs: list[dict], cfg: dict[str, Any]) -> list[dict]:
    """Flatten grounding pairs into question instances (correct image + partner image)."""
    p = cfg.get("prompts", {})
    out = []
    for pair in pairs:
        if pair["type"] == "presence":
            q = p.get("presence_template", "Does this image contain {object}?").format(object=pair["object"])
            out.append(_inst(pair_id=_gpid(pair), suite="grounding", subtype="presence", question=q,
                             target="yes", correct=pair["image_has"], partner=pair["image_lacks"]))
            out.append(_inst(pair_id=_gpid(pair), suite="grounding", subtype="presence", question=q,
                             target="no", correct=pair["image_lacks"], partner=pair["image_has"]))
        else:
            q = p.get("counting_template", "How many {object} can you see?").format(object=pair["object"])
            out.append(_inst(pair_id=_gpid(pair), suite="grounding", subtype="counting", question=q,
                             target=str(pair["count_a"]), correct=pair["image_a"], partner=pair["image_b"]))
            out.append(_inst(pair_id=_gpid(pair), suite="grounding", subtype="counting", question=q,
                             target=str(pair["count_b"]), correct=pair["image_b"], partner=pair["image_a"]))
    return out


def _natural_instances(pairs: list[dict]) -> list[dict]:
    out = []
    for pair in pairs:
        q0, q1 = pair["questions"]
        for a, b in ((q0, q1), (q1, q0)):
            out.append(_inst(pair_id=pair["pair_id"], suite="natural_vqa",
                             subtype=_vqa_type(a.get("question_type", "other")),
                             question=a["question"], target=a["target"], answers=a.get("answers", []),
                             correct=a["image_id"], partner=b["image_id"]))
    return out


def _inst(**kw) -> dict:
    kw.setdefault("answers", [])
    return kw


def _gpid(pair: dict) -> str:
    if pair["type"] == "presence":
        return f"presence:{pair['category_id']}:{pair['image_has']}:{pair['image_lacks']}"
    return f"counting:{pair['category_id']}:{pair['image_a']}:{pair['image_b']}"


def _vqa_type(t: str) -> str:
    return {"yes/no": "yes_no", "number": "number", "other": "other"}.get(t, "other")


# -- NLL + generation primitives -------------------------------------------------------------------

def _example(model: SigLIPQwenVLM, cfg: dict[str, Any], image_id: int, question: str,
             answer: str, *, caption: bool = False):
    q_cap, cap_cap, ans_cap = _caps(cfg)
    f = format_vlm_example(model.tokenizer, question, answer,
                           answer_cap=cap_cap if caption else ans_cap, question_cap=q_cap,
                           enable_thinking=model.enable_thinking)
    from .data import VLMExample
    return VLMExample(image_id=int(image_id), component="eval", input_ids=f["input_ids"],
                      labels=f["labels"], insert_at=f["insert_at"],
                      n_answer_tokens=f["n_answer_tokens"], target=answer)


def _features(model: SigLIPQwenVLM, cfg: dict[str, Any], examples) -> torch.Tensor:
    return torch.stack([load_cached_features(cfg, ex.image_id, device=model.device) for ex in examples])


@torch.no_grad()
def _answer_nlls(model: SigLIPQwenVLM, cfg: dict[str, Any], examples, *, with_image: bool,
                 exclude_termination: bool, pcfg, amp) -> list[float]:
    """Per-example teacher-forced answer NLL (mean over answer tokens, optional EOT exclusion)."""
    visual = model._visual_embeds(_features(model, cfg, examples), pcfg, amp) if with_image else None
    embeds, attn, pos, labels = model.build_inputs(examples, visual, with_image=with_image)
    from common.precision import autocast_context
    with autocast_context(pcfg, device_type=model.device.type, enabled=amp):
        logits = model.qwen(inputs_embeds=embeds, attention_mask=attn, position_ids=pos).logits
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    B, Tm1, V = shift_logits.shape
    nll = F.cross_entropy(shift_logits.reshape(-1, V).float(), shift_labels.reshape(-1),
                          ignore_index=IGNORE_INDEX, reduction="none").view(B, Tm1)
    mask = shift_labels != IGNORE_INDEX
    out = []
    for b in range(B):
        idx = mask[b].nonzero().flatten()
        if exclude_termination and idx.numel() > 0:
            idx = idx[:-1]                               # drop the end-of-turn token
        out.append(float(nll[b, idx].mean()) if idx.numel() else float("nan"))
    return out


def reference_nll(model, batch, *, exclude_termination: bool = True) -> float:
    """Teacher-forced per-example reference NLL for one (example, cfg) batch tuple."""
    cfg, examples, with_image, pcfg, amp = batch
    vals = _answer_nlls(model, cfg, examples, with_image=with_image,
                        exclude_termination=exclude_termination, pcfg=pcfg, amp=amp)
    return float(np.nanmean(vals)) if vals else float("nan")


def delta_nll(model, pair, *, condition_correct, condition_partner) -> float:
    """delta_NLL = NLL(y|q, I_partner) - NLL(y|q, I_correct) for one instance (p8)."""
    return condition_partner - condition_correct


# -- development -----------------------------------------------------------------------------------

def _precision(cfg):
    from common.precision import PrecisionConfig
    return PrecisionConfig(compute_dtype=cfg.get("precision", {}).get("compute_dtype", "float16"),
                           fp32_master=False, grad_clip_norm=1.0)


def run_development(model: SigLIPQwenVLM, cfg: dict[str, Any], update: int) -> dict[str, Any]:
    """Dev metrics at a checkpoint: caption loss, reference-answer NLL, delta_NLL per condition."""
    pcfg = _precision(cfg)
    amp = model.device.type == "cuda"
    bs = int(cfg.get("eval", {}).get("dev_batch_size", 8))
    model.eval()

    metrics: dict[str, Any] = {}

    # Caption dev reference loss (mean answer NLL over caption dev examples).
    cap_dev = _load(cfg, "captions_dev.json")
    if cap_dev:
        exs = [_example(model, cfg, c["image_id"], cfg.get("prompts", {})
                        .get("caption_instruction", "Describe this image."), c["caption"], caption=True)
               for c in cap_dev]
        metrics["caption_dev_nll"] = _batched_mean_nll(model, cfg, exs, True, pcfg, amp, bs)

    # Both VQA dev suites under correct / complementary / no-image + delta_NLL.
    suites = {"grounding": _grounding_instances(_load(cfg, "grounding_dev.json"), cfg),
              "natural_vqa": _natural_instances(_load(cfg, "natural_vqa_dev.json"))}
    for suite, instances in suites.items():
        if not instances:
            continue
        corr = _instances_nll(model, cfg, instances, "correct", pcfg, amp, bs)
        comp = _instances_nll(model, cfg, instances, "partner", pcfg, amp, bs)
        noimg = _instances_nll(model, cfg, instances, "no_image", pcfg, amp, bs)
        metrics[f"{suite}_nll_correct"] = float(np.nanmean(corr))
        metrics[f"{suite}_nll_complementary"] = float(np.nanmean(comp))
        metrics[f"{suite}_nll_no_image"] = float(np.nanmean(noimg))
        metrics[f"{suite}_delta_nll"] = float(np.nanmean(np.array(comp) - np.array(corr)))

    model.train()
    return metrics


def _instances_nll(model, cfg, instances, condition, pcfg, amp, bs) -> list[float]:
    """Per-instance answer NLL under an image condition (correct|partner|no_image)."""
    with_image = condition != "no_image"
    out: list[float] = []
    for i in range(0, len(instances), bs):
        chunk = instances[i:i + bs]
        exs = [_example(model, cfg, _cond_image(inst, condition), inst["question"], inst["target"])
               for inst in chunk]
        out.extend(_answer_nlls(model, cfg, exs, with_image=with_image,
                                exclude_termination=True, pcfg=pcfg, amp=amp))
    return out


def _cond_image(inst: dict, condition: str) -> int:
    if condition == "partner":
        return inst["partner"]
    if condition == "mismatched":
        return inst["mismatched"]
    return inst["correct"]                               # correct or no_image (image unused when no_image)


def _batched_mean_nll(model, cfg, examples, with_image, pcfg, amp, bs) -> float:
    vals = []
    for i in range(0, len(examples), bs):
        vals.extend(_answer_nlls(model, cfg, examples[i:i + bs], with_image=with_image,
                                 exclude_termination=True, pcfg=pcfg, amp=amp))
    return float(np.nanmean(vals)) if vals else float("nan")


# -- final evaluation ------------------------------------------------------------------------------

def _load_final_model(cfg: dict[str, Any]) -> SigLIPQwenVLM:
    """Build the VLM and load the final (latest/Stage-3) connector+adapter weights."""
    run_id = cfg.get("run_id", "task2_run")
    ckpt_dir = os.fspath(task2_dir("checkpoints") / run_id)
    latest = latest_checkpoint(ckpt_dir)
    if latest is None:
        raise SystemExit(f"No Task-2 checkpoint found in {ckpt_dir}; train first.")
    model = SigLIPQwenVLM(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    from .train import _load_weights
    _load_weights(model, load_checkpoint(latest).weights)
    model.eval()
    return model


def _mismatched_mapping(cfg: dict[str, Any], instances: list[dict], suite: str,
                        eval_cfg: dict[str, Any]) -> dict[str, int]:
    """Fix, per instance, an unrelated image from another pair in the same suite (excludes the
    original pair's two images). Reused across comparisons; cached to disk (p8)."""
    path = os.fspath(REPO_ROOT / eval_cfg.get("final", {})
                     .get("mismatched_mapping_file", "splits/task2/mismatched_mapping.json"))
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    key = suite
    if key in cache:
        mapping = cache[key]
        return {k: int(v) for k, v in mapping.items()}

    rng = np.random.default_rng(SEED)
    all_images = [inst["correct"] for inst in instances]
    mapping: dict[str, int] = {}
    for idx, inst in enumerate(instances):
        excluded = {inst["correct"], inst["partner"]}
        choices = [img for img in all_images if img not in excluded]
        mapping[str(idx)] = int(rng.choice(choices)) if choices else int(inst["correct"])
    cache[key] = mapping
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return mapping


@torch.no_grad()
def run_final(cfg: dict[str, Any]) -> dict[str, Any]:
    """Final-test evaluation under the four image conditions; score + bootstrap; write results."""
    eval_cfg = load_yaml(cfg["eval"]["config"]) if "config" in cfg.get("eval", {}) else cfg
    final = eval_cfg.get("final", {})
    max_new = int(final.get("max_new_tokens", 16))
    bs = int(final.get("batch_size", 8))
    pcfg = _precision(cfg)
    amp = torch.cuda.is_available()

    model = _load_final_model(cfg)

    suites = {"grounding": _grounding_instances(_load(cfg, "grounding_test.json"), cfg),
              "natural_vqa": _natural_instances(_load(cfg, "natural_vqa_test.json"))}
    conditions = ["correct", "complementary_swap", "mismatched", "no_image"]
    results: dict[str, Any] = {}

    for suite, instances in suites.items():
        if not instances:
            continue
        mism = _mismatched_mapping(cfg, instances, suite, eval_cfg)
        for idx, inst in enumerate(instances):
            inst["mismatched"] = mism[str(idx)]

        suite_res: dict[str, Any] = {}
        for cond in conditions:
            preds, nlls, times = _generate_condition(model, cfg, instances, cond, max_new, bs, pcfg, amp)
            suite_res[cond] = _score_condition(instances, preds, nlls, times, suite)
        results[suite] = suite_res
        # Correct-image advantages (accuracy + NLL) with pair bootstrap.
        results[f"{suite}_advantage"] = _correct_image_advantage(
            instances, suite_res, int(eval_cfg.get("scoring", {}).get("bootstrap_resamples", 4000)))

    out_dir = ensure_dir(REPO_ROOT / "results" / "task2")
    with open(os.fspath(out_dir / f"{cfg.get('run_id', 'task2_run')}_final.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def _cond_key(cond: str) -> str:
    return {"complementary_swap": "partner"}.get(cond, cond)


def _generate_condition(model, cfg, instances, cond, max_new, bs, pcfg, amp):
    """Return (predictions, reference NLLs, per-answer gen times) for one image condition."""
    ck = _cond_key(cond)
    with_image = cond != "no_image"
    preds, nlls, times = [], [], []
    for i in range(0, len(instances), bs):
        chunk = instances[i:i + bs]
        gen_exs = [_example(model, cfg, _cond_image(inst, ck), inst["question"], "") for inst in chunk]
        t0 = time.perf_counter()
        out = model.generate(gen_exs, _features(model, cfg, gen_exs) if with_image else gen_exs,
                             max_new_tokens=max_new, pcfg=pcfg, amp_enabled=amp,
                             with_image=with_image, greedy=True)
        dt = (time.perf_counter() - t0) / max(1, len(chunk))
        preds.extend(out)
        times.extend([dt] * len(chunk))
        ref_exs = [_example(model, cfg, _cond_image(inst, ck), inst["question"], inst["target"])
                   for inst in chunk]
        nlls.extend(_answer_nlls(model, cfg, ref_exs, with_image=with_image,
                                 exclude_termination=True, pcfg=pcfg, amp=amp))
    return preds, nlls, times


def _score_condition(instances, preds, nlls, times, suite) -> dict[str, Any]:
    """Accuracy (per subtype), pair accuracy, identical/empty rates, mean NLL, gen time."""
    per_type: dict[str, list[float]] = {}
    correct_by_pair: dict[str, list[bool]] = {}
    empty = 0
    for inst, pred in zip(instances, preds):
        if suite == "natural_vqa":
            score = soft_consensus_accuracy(pred, inst["answers"]) if inst["answers"] \
                else normalized_exact_match(pred, inst["target"])
        else:
            score = normalized_exact_match(pred, inst["target"])
        per_type.setdefault(inst["subtype"], []).append(score)
        correct_by_pair.setdefault(inst["pair_id"], []).append(score >= 0.5)
        if normalize_answer(pred) == "":
            empty += 1

    pair_acc = float(np.mean([all(v) for v in correct_by_pair.values()])) if correct_by_pair else 0.0
    identical = _identical_answer_rate(instances, preds)
    return {
        "accuracy_overall": float(np.mean([s for v in per_type.values() for s in v])) if per_type else 0.0,
        "accuracy_by_type": {k: float(np.mean(v)) for k, v in per_type.items()},
        "pair_accuracy": pair_acc,
        "identical_answer_rate": identical,
        "empty_answer_rate": empty / max(1, len(preds)),
        "mean_reference_nll": float(np.nanmean(nlls)) if nlls else float("nan"),
        "gen_seconds_per_answer": float(np.mean(times)) if times else 0.0,
        "_per_instance_scores": [s for v in per_type.values() for s in v],  # internal (bootstrap)
    }


def _identical_answer_rate(instances, preds) -> float:
    """Fraction of pairs whose two generated answers normalize to the same string."""
    by_pair: dict[str, list[str]] = {}
    for inst, pred in zip(instances, preds):
        by_pair.setdefault(inst["pair_id"], []).append(normalize_answer(pred))
    both = [v for v in by_pair.values() if len(v) == 2]
    return float(np.mean([a == b for a, b in both])) if both else 0.0


def _correct_image_advantage(instances, suite_res, resamples) -> dict[str, Any]:
    """Correct-vs-complementary advantage in accuracy + NLL with pair bootstrap CIs."""
    # Per-pair correct/complementary accuracy and NLL, aligned by pair id.
    pair_ids = sorted({inst["pair_id"] for inst in instances})
    corr = suite_res["correct"]
    comp = suite_res["complementary_swap"]
    acc_diff = np.array(corr["_per_instance_scores"]) - np.array(comp["_per_instance_scores"])
    # NLL advantage: complementary NLL - correct NLL (higher => correct image helps).
    nll_adv = comp["mean_reference_nll"] - corr["mean_reference_nll"]
    acc_ci = bootstrap_pair_difference(acc_diff.tolist(), resamples=resamples)
    return {"accuracy_advantage_mean": float(acc_diff.mean()) if acc_diff.size else 0.0,
            "accuracy_advantage_ci95": acc_ci, "nll_advantage": float(nll_adv),
            "n_pairs": len(pair_ids)}


def bootstrap_pair_difference(per_pair_values, resamples: int = 4000, *, seed: int = SEED):
    """95% CI for a correct-vs-complementary difference by resampling complete pairs. (p9)"""
    vals = np.asarray(per_pair_values, dtype=np.float64)
    if vals.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    n = vals.size
    for b in range(resamples):
        means[b] = vals[rng.integers(0, n, n)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    from common.config import build_arg_parser
    parser = build_arg_parser("Task 2 grounding evaluation (development/final).")
    parser.add_argument("--mode", choices=["final"], default="final",
                        help="Evaluation mode (development runs inside training).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    cfg.setdefault("run_id", "task2_run")
    res = run_final(cfg)
    print("[task2.evaluate] final results written. Summary:")
    for suite, r in res.items():
        if isinstance(r, dict) and "correct" in r:
            print(f"  {suite}: correct acc={r['correct']['accuracy_overall']:.3f} "
                  f"complementary acc={r['complementary_swap']['accuracy_overall']:.3f}")


if __name__ == "__main__":
    main()
