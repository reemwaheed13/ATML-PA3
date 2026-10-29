"""Task 3 live generation: three strategies on 50 MATH questions. (handout p12-p13, Step 4)

Qwen2.5-3B-Instruct (revision from the data README), FP16, batch 1, temperature 0.7 / top-p 0.8 /
top-k 20 / repetition penalty 1.0, native chat template, system "You are a helpful assistant.",
problem + "Think step by step inside <think> </think> tags, then give your final answer inside
\\boxed{}." Extract the last boxed answer via the balanced-brace rule in the README.

Three strategies under a 3,072-new-token total output allowance per question:
  * single_long: one response capped at 3,072.
  * parallel: three independent responses capped at 1,024 each; self-consistency selects.
  * sequential_refinement: reuse the first parallel response (cap 1,024) as the initial, then one
    critique-and-revision response capped at 2,048 (no gold / no correctness feedback).

Keep weights fixed; save prompts, responses, seeds, lengths, truncation flags. Measure
end-to-end inference time (excluding model loading), input/output tokens, and peak GPU memory.
Report wrong-to-correct / correct-to-wrong transitions vs the shared initial answer. These FP16
runs are a NEW controlled comparison, not a reproduction of the supplied samples.

NOTE: the problem text + gold for the live-50 come from `questions.source` (records with
sample_id/problem/gold); prepare it from the MATH data per README.MD. The extraction rule + model
revision are config fields pending README.MD confirmation.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any

import torch

from common import SEED
from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, ensure_dir
from . import data as t3data

NO_ANSWER = "NO_ANSWER"


def extract_last_boxed(text: str) -> str:
    """Extract the last \\boxed{...} answer using a balanced-brace scan (README rule)."""
    token = "\\boxed"
    results: list[str] = []
    idx = 0
    while True:
        j = text.find(token, idx)
        if j == -1:
            break
        k = j + len(token)
        while k < len(text) and text[k] == " ":
            k += 1
        if k >= len(text) or text[k] != "{":
            idx = j + len(token)
            continue
        depth, p = 0, k
        while p < len(text):
            if text[p] == "{":
                depth += 1
            elif text[p] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[k + 1:p])
                    break
            p += 1
        idx = p + 1
    return results[-1].strip() if results else NO_ANSWER


# =================================================================================================
# Model + generation
# =================================================================================================

def _load_model(cfg: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mc = cfg["model"]
    kw = {"dtype": torch.float16}
    if mc.get("revision"):
        kw["revision"] = mc["revision"]
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], revision=mc.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(mc["name"], **kw)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def _generate(model, tokenizer, device, messages, *, max_new_tokens: int, sampling: dict,
              seed: int) -> dict[str, Any]:
    """One chat-templated generation; returns text, token counts, truncation flag, seed."""
    torch.manual_seed(seed)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(device)
    gen = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=float(sampling.get("temperature", 0.7)),
        top_p=float(sampling.get("top_p", 0.8)), top_k=int(sampling.get("top_k", 20)),
        repetition_penalty=float(sampling.get("repetition_penalty", 1.0)),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    out_ids = gen[0][input_ids.shape[1]:]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    truncated = int(out_ids.shape[0]) >= max_new_tokens      # hit cap without terminating
    return {"text": text, "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(out_ids.shape[0]), "truncated": bool(truncated), "seed": seed}


def _messages(cfg: dict[str, Any], problem: str) -> list[dict]:
    p = cfg["prompt"]
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": f"{problem}\n{p['instruction']}"}]


# =================================================================================================
# Strategies
# =================================================================================================

def run_single_long(model, tokenizer, device, question, cfg, seed) -> dict[str, Any]:
    """One response capped at 3,072 new tokens."""
    cap = int(cfg["strategies"]["single_long"]["caps"][0])
    g = _generate(model, tokenizer, device, _messages(cfg, question["problem"]),
                  max_new_tokens=cap, sampling=cfg["sampling"], seed=seed)
    g["answer"] = extract_last_boxed(g["text"])
    return g


def run_parallel(model, tokenizer, device, question, cfg, seed) -> dict[str, Any]:
    """Three independent responses (cap 1,024 each); self-consistency selects the answer."""
    st = cfg["strategies"]["parallel"]
    caps = st["caps"]
    responses = [_generate(model, tokenizer, device, _messages(cfg, question["problem"]),
                           max_new_tokens=int(caps[i]), sampling=cfg["sampling"], seed=seed + i)
                 for i in range(int(st["responses"]))]
    answers = [extract_last_boxed(r["text"]) for r in responses]
    selected = _self_consistency(answers, question["dataset"])
    return {"responses": responses, "answers": answers, "answer": selected,
            "initial_answer": answers[0],
            "input_tokens": sum(r["input_tokens"] for r in responses),
            "output_tokens": sum(r["output_tokens"] for r in responses),
            "truncated": any(r["truncated"] for r in responses)}


def run_sequential_refinement(model, tokenizer, device, question, initial, cfg, seed) -> dict[str, Any]:
    """Critique-and-revision (cap 2,048) conditioned on the reused initial response."""
    st = cfg["strategies"]["sequential_refinement"]
    messages = _messages(cfg, question["problem"]) + [
        {"role": "assistant", "content": initial["text"]},
        {"role": "user", "content": st["revision_instruction"]},
    ]
    g = _generate(model, tokenizer, device, messages, max_new_tokens=int(st["revision_cap"]),
                  sampling=cfg["sampling"], seed=seed + 100)
    g["answer"] = extract_last_boxed(g["text"])
    g["initial_answer"] = initial["answer"]
    # Charge both the shared initial response and the revision prompt (p13).
    g["input_tokens"] = g["input_tokens"]
    g["output_tokens"] = initial["output_tokens"] + g["output_tokens"]
    return g


def _self_consistency(answers: list[str], dataset: str) -> str:
    """Majority answer by math equivalence; ties -> earliest; NO_ANSWER only if all missing."""
    reps: list[str] = []
    votes: list[int] = []
    assign: list[int] = []
    for a in answers:
        if a == NO_ANSWER:
            assign.append(-1)
            continue
        gid = None
        for i, rep in enumerate(reps):
            if t3data.equivalence(a, rep, dataset):
                gid = i
                break
        if gid is None:
            reps.append(a); votes.append(0); gid = len(reps) - 1
        votes[gid] += 1
        assign.append(gid)
    if not reps:
        return NO_ANSWER
    best = max(votes)
    for a, gid in zip(answers, assign):           # earliest supporting answer among tied leaders
        if gid >= 0 and votes[gid] == best:
            return a
    return reps[0]


# =================================================================================================
# Run
# =================================================================================================

def _load_questions(cfg: dict[str, Any]) -> list[dict]:
    path = os.fspath(REPO_ROOT / cfg["questions"]["source"])
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw:
        out.append({"sample_id": str(r.get("sample_id", r.get("id"))),
                    "problem": r.get("problem") or r.get("question"),
                    "gold": r.get("gold") or r.get("answer"),
                    "dataset": r.get("dataset", "math")})
    return out


def run(cfg: dict[str, Any]) -> None:
    """Run all three strategies over the 50 live MATH questions; save outputs + resource stats."""
    questions = _load_questions(cfg)
    model, tokenizer, device = _load_model(cfg)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir = ensure_dir(REPO_ROOT / "results" / "task3" / "live")
    records: list[dict] = []
    t0 = time.perf_counter()
    for qi, q in enumerate(questions):
        seed = SEED + qi
        single = run_single_long(model, tokenizer, device, q, cfg, seed)
        parallel = run_parallel(model, tokenizer, device, q, cfg, seed)
        initial = {"text": parallel["responses"][0]["text"], "answer": parallel["answers"][0],
                   "output_tokens": parallel["responses"][0]["output_tokens"]}
        refine = run_sequential_refinement(model, tokenizer, device, q, initial, cfg, seed)

        gold = q["gold"]
        rec = {
            "sample_id": q["sample_id"],
            "gold": gold,
            "single_long": _summ(single, gold, q["dataset"]),
            "parallel": _summ(parallel, gold, q["dataset"], initial_answer=parallel["initial_answer"]),
            "sequential_refinement": _summ(refine, gold, q["dataset"],
                                           initial_answer=refine["initial_answer"]),
        }
        records.append(rec)
        print(f"[task3.live] {qi + 1}/{len(questions)} {q['sample_id']} "
              f"single={rec['single_long']['correct']} par={rec['parallel']['correct']} "
              f"refine={rec['sequential_refinement']['correct']}", flush=True)

    wall = time.perf_counter() - t0
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    summary = _aggregate(records, wall, peak)

    with open(os.fspath(out_dir / "live_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(os.fspath(out_dir / "live_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[task3.live] wrote {len(records)} records; {wall:.1f}s, peak GPU {peak} bytes.")


def _summ(res: dict, gold, dataset: str, *, initial_answer=None) -> dict[str, Any]:
    answer = res["answer"]
    correct = gold is not None and answer != NO_ANSWER and t3data.equivalence(answer, gold, dataset)
    out = {"answer": answer, "correct": bool(correct), "truncated": bool(res.get("truncated")),
           "extracted": answer != NO_ANSWER, "input_tokens": res.get("input_tokens"),
           "output_tokens": res.get("output_tokens")}
    if initial_answer is not None:
        init_correct = (gold is not None and initial_answer != NO_ANSWER
                        and t3data.equivalence(initial_answer, gold, dataset))
        out["initial_correct"] = bool(init_correct)
        out["wrong_to_correct"] = bool(correct and not init_correct)
        out["correct_to_wrong"] = bool(init_correct and not correct)
    return out


def _aggregate(records: list[dict], wall: float, peak: int) -> dict[str, Any]:
    strategies = ["single_long", "parallel", "sequential_refinement"]
    summary: dict[str, Any] = {"n_questions": len(records), "wall_seconds": wall,
                               "peak_gpu_bytes": peak}
    for s in strategies:
        rows = [r[s] for r in records]
        agg = {
            "accuracy": _mean([r["correct"] for r in rows]),
            "truncation_rate": _mean([r["truncated"] for r in rows]),
            "extraction_rate": _mean([r["extracted"] for r in rows]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rows]),
            "mean_input_tokens": _mean([r["input_tokens"] for r in rows]),
        }
        if s != "single_long":
            agg["wrong_to_correct"] = sum(r.get("wrong_to_correct", False) for r in rows)
            agg["correct_to_wrong"] = sum(r.get("correct_to_wrong", False) for r in rows)
        summary[s] = agg
    return summary


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    parser = build_arg_parser("Task 3 live generation (three strategies on 50 MATH questions).")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
