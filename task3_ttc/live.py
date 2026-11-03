"""Task 3 live generation: three strategies on 50 MATH questions. (handout p12-p13, Step 4)

Qwen2.5-3B-Instruct (revision from the data README), FP16, batch 1, temperature 0.7 / top-p 0.8 /
top-k 20 / repetition penalty 1.0, native chat template, system "You are a helpful assistant.",
problem + "Think step by step inside <think> </think> tags, then give your final answer inside
\\boxed{}." Extract the last boxed answer via the balanced-brace rule.

Strategies under a 3,072-new-token total allowance per question: single_long (one cap-3072
response); parallel (three cap-1024 responses, self-consistency selects — SAME rule as offline:
NO_ANSWER is its own group and can win, ties to the earliest); sequential_refinement (reuse the
first parallel response as the initial, then one cap-2048 critique/revision).

Audit fixes: deterministic per-call seeds (base = 6304*1000 + 10*qi; single=+0, parallel=+1/+2/+3,
revision=+4), incremental + resumable JSONL, full chat-templated prompt + response + tokens +
truncation + seed + per-call wall time (cuda-synchronized) stored for every call, per-question and
overall peak GPU memory. live_summary.json is rebuilt from the JSONL at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any

import numpy as np
import torch

from common.config import build_arg_parser, load_yaml
from common.paths import REPO_ROOT, ensure_dir, task3_dir
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
    rev = mc.get("revision")
    tokenizer = AutoTokenizer.from_pretrained(mc["name"], revision=rev)
    # transformers 5.x takes `dtype=`; keep the pinned behaviour (requirements.txt: transformers 5.x).
    model = AutoModelForCausalLM.from_pretrained(mc["name"], revision=rev, dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _generate(model, tokenizer, device, messages, *, max_new_tokens: int, sampling: dict,
              seed: int) -> dict[str, Any]:
    """One chat-templated generation; records prompt, response, tokens, truncation, seed, wall time."""
    torch.manual_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt").to(device)
    _sync(device)
    t0 = time.perf_counter()
    gen = model.generate(
        input_ids, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=float(sampling.get("temperature", 0.7)),
        top_p=float(sampling.get("top_p", 0.8)), top_k=int(sampling.get("top_k", 20)),
        repetition_penalty=float(sampling.get("repetition_penalty", 1.0)),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    _sync(device)
    dt = time.perf_counter() - t0
    out_ids = gen[0][input_ids.shape[1]:]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    answer = extract_last_boxed(text)
    return {"prompt": prompt, "text": text, "answer": answer,
            "input_tokens": int(input_ids.shape[1]), "output_tokens": int(out_ids.shape[0]),
            "truncated": bool(int(out_ids.shape[0]) >= max_new_tokens), "seed": int(seed),
            "time": float(dt)}


def _messages(cfg: dict[str, Any], problem: str) -> list[dict]:
    p = cfg["prompt"]
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": f"{problem}\n{p['instruction']}"}]


def _sc_select(answers: list[str], dataset: str) -> tuple[str, int]:
    """Self-consistency (offline rule): NO_ANSWER is its own group and can win; ties -> earliest."""
    groups = t3data.answer_groups(answers, dataset, None)
    g = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(g)
    best = counts.max()
    win = int(g[int(np.argmax(counts[g] == best))])
    rep = t3data.group_representatives(answers, groups)
    return rep.get(win, NO_ANSWER), win


# =================================================================================================
# Strategies (one question)
# =================================================================================================

def _run_question(model, tokenizer, device, q, cfg, base_seed: int) -> dict[str, Any]:
    dataset = q["dataset"]
    gold = q["gold"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    st = cfg["strategies"]
    single = _generate(model, tokenizer, device, _messages(cfg, q["problem"]),
                       max_new_tokens=int(st["single_long"]["caps"][0]), sampling=cfg["sampling"],
                       seed=base_seed + 0)

    pcaps = st["parallel"]["caps"]
    par = [_generate(model, tokenizer, device, _messages(cfg, q["problem"]),
                     max_new_tokens=int(pcaps[i]), sampling=cfg["sampling"], seed=base_seed + 1 + i)
           for i in range(int(st["parallel"]["responses"]))]
    par_answers = [r["answer"] for r in par]
    par_selected, _ = _sc_select(par_answers, dataset)
    initial = par[0]

    rst = st["sequential_refinement"]
    rev_messages = _messages(cfg, q["problem"]) + [
        {"role": "assistant", "content": initial["text"]},
        {"role": "user", "content": rst["revision_instruction"]},
    ]
    revision = _generate(model, tokenizer, device, rev_messages,
                         max_new_tokens=int(rst["revision_cap"]), sampling=cfg["sampling"],
                         seed=base_seed + 4)

    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0

    def correct(ans):
        return bool(gold is not None and ans != NO_ANSWER and t3data.equivalence(ans, gold, dataset))

    init_ans = par_answers[0]
    rec = {
        "sample_id": q["sample_id"], "dataset": dataset, "gold": gold,
        "seeds": {"single_long": base_seed + 0, "parallel": [base_seed + 1, base_seed + 2,
                                                             base_seed + 3],
                  "sequential_refinement": base_seed + 4},
        "peak_gpu_bytes": peak,
        "single_long": {
            "calls": [single], "answer": single["answer"], "correct": correct(single["answer"]),
            "input_tokens": single["input_tokens"], "output_tokens": single["output_tokens"],
            "truncated": single["truncated"], "time": single["time"]},
        "parallel": {
            "calls": par, "answers": par_answers, "answer": par_selected,
            "correct": correct(par_selected), "initial_answer": init_ans,
            "initial_correct": correct(init_ans),
            "wrong_to_correct": correct(par_selected) and not correct(init_ans),
            "correct_to_wrong": correct(init_ans) and not correct(par_selected),
            "input_tokens": sum(r["input_tokens"] for r in par),
            "output_tokens": sum(r["output_tokens"] for r in par),
            "per_response_truncation": [r["truncated"] for r in par],
            "truncated": any(r["truncated"] for r in par),
            "time": sum(r["time"] for r in par)},
        "sequential_refinement": {
            "initial": initial, "revision": revision, "answer": revision["answer"],
            "correct": correct(revision["answer"]), "initial_answer": init_ans,
            "initial_correct": correct(init_ans),
            "wrong_to_correct": correct(revision["answer"]) and not correct(init_ans),
            "correct_to_wrong": correct(init_ans) and not correct(revision["answer"]),
            "input_tokens": initial["input_tokens"] + revision["input_tokens"],
            "output_tokens": initial["output_tokens"] + revision["output_tokens"],
            "truncated_initial": initial["truncated"], "truncated_revision": revision["truncated"],
            "truncated": bool(initial["truncated"] or revision["truncated"]),
            "time": initial["time"] + revision["time"]},
    }
    return rec


# =================================================================================================
# Run (incremental JSONL, resumable) + summary
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


def _done_ids(path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(str(json.loads(line)["sample_id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run(cfg: dict[str, Any]) -> None:
    questions = _load_questions(cfg)
    live_dir = ensure_dir(task3_dir("live"))
    jsonl_path = os.fspath(live_dir / "live_records.jsonl")
    done = _done_ids(jsonl_path)
    todo = [q for q in questions if q["sample_id"] not in done]
    print(f"[task3.live] {len(done)} already done; running {len(todo)} of {len(questions)}.")

    model = tokenizer = device = None
    if todo:
        model, tokenizer, device = _load_model(cfg)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for q in todo:
                qi = questions.index(q)
                base = 6304 * 1000 + 10 * qi
                rec = _run_question(model, tokenizer, device, q, cfg, base)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[task3.live] {q['sample_id']} single={rec['single_long']['correct']} "
                      f"par={rec['parallel']['correct']} "
                      f"refine={rec['sequential_refinement']['correct']}", flush=True)

    summary = _summarize(jsonl_path)
    summary_path = os.fspath(live_dir / "live_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    res_dir = ensure_dir(REPO_ROOT / "results" / "task3" / "live")
    shutil.copy(jsonl_path, os.fspath(res_dir / "live_records.jsonl"))
    shutil.copy(summary_path, os.fspath(res_dir / "live_summary.json"))
    print(f"[task3.live] wrote summary for {summary['n_questions']} questions -> {summary_path}")


def _summarize(jsonl_path: str) -> dict[str, Any]:
    recs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    strategies = ["single_long", "parallel", "sequential_refinement"]
    summary: dict[str, Any] = {"n_questions": len(recs),
                               "peak_gpu_bytes": max((int(r.get("peak_gpu_bytes", 0)) for r in recs),
                                                     default=0)}
    for s in strategies:
        rows = [r[s] for r in recs]
        agg = {
            "accuracy": _mean([r["correct"] for r in rows]),
            "extraction_rate": _mean([r["answer"] != NO_ANSWER for r in rows]),
            "truncation_rate": _mean([bool(r.get("truncated", r.get("truncated_revision", False)))
                                      for r in rows]),
            "mean_input_tokens": _mean([r["input_tokens"] for r in rows]),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rows]),
            "total_input_tokens": int(sum(r["input_tokens"] for r in rows)),
            "total_output_tokens": int(sum(r["output_tokens"] for r in rows)),
            "total_runtime_s": float(sum(r["time"] for r in rows)),
            "mean_runtime_s": _mean([r["time"] for r in rows]),
        }
        if s == "parallel":
            n_resp = len(rows[0]["per_response_truncation"]) if rows else 0
            agg["per_response_truncation_rate"] = [
                _mean([r["per_response_truncation"][i] for r in rows]) for i in range(n_resp)]
            agg["wrong_to_correct"] = int(sum(r.get("wrong_to_correct", False) for r in rows))
            agg["correct_to_wrong"] = int(sum(r.get("correct_to_wrong", False) for r in rows))
        if s == "sequential_refinement":
            agg["truncation_rate_initial"] = _mean([r["truncated_initial"] for r in rows])
            agg["truncation_rate_revision"] = _mean([r["truncated_revision"] for r in rows])
            agg["wrong_to_correct"] = int(sum(r.get("wrong_to_correct", False) for r in rows))
            agg["correct_to_wrong"] = int(sum(r.get("correct_to_wrong", False) for r in rows))
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
