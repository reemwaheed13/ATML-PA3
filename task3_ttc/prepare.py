"""Prepare Task 3 splits, per-question orders, and the live-50 question file. (handout p11)

`python -m task3_ttc.prepare --config configs/task3/offline.yaml`

Writes, under splits/task3/ (committed, write-once):
  * <dataset>_splits.json  — dev/eval IDs in permutation order (make_splits + save_splits).
  * <dataset>_orders.json  — ten seeded rollout-id permutations per question (offline helper).
  * math_live_50.json      — [{sample_id, problem, gold, dataset:"math"}] for the first 50 MATH
                             eval questions in permutation order; gold is the supplied gold_answer.

Problem text for the live file is read from the MATH generations.parquet; we look for a column among
PROBLEM_COLUMNS. If none is present we dump the README.MD lines that mention problem/MATH/source/
revision and raise, rather than guessing an external dataset source.
"""

from __future__ import annotations

import json
import os
from typing import Any

from common.config import build_arg_parser, load_yaml
from common.paths import SPLITS_DIR, ensure_dir
from . import data as t3data
from .offline import build_question_arrays, load_or_make_orders
from .splits import live_math_50, make_splits, save_splits

PROBLEM_COLUMNS = ["problem", "question", "question_text", "prompt", "input", "query"]


def _math_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    for spec in cfg["data"]["models"]:
        if str(spec["dataset"]) == "math":
            return spec
    raise ValueError("no model with dataset == 'math' in the offline config")


def _find_problem_column(gen_path: str) -> str:
    import pyarrow.parquet as pq
    names = pq.ParquetFile(gen_path).schema_arrow.names
    for cand in PROBLEM_COLUMNS:
        if cand in names:
            return cand
    return ""


def _dump_readme_and_raise(cfg: dict[str, Any], gen_path: str) -> None:
    root = os.path.expandvars(cfg["data"]["root"])
    readme = os.path.expandvars(cfg["data"].get("readme", os.path.join(root, "README.MD")))
    print(f"[prepare] no problem-text column (tried {PROBLEM_COLUMNS}) in {gen_path}.")
    try:
        with open(readme, encoding="utf-8", errors="replace") as f:
            for line in f:
                low = line.lower()
                if any(k in low for k in ("problem", "math", "source", "revision")):
                    print("  README:", line.rstrip())
    except OSError as exc:
        print(f"[prepare] could not read README at {readme}: {exc}")
    raise RuntimeError(
        "Live-50 needs the original MATH problem text. It is not in generations.parquet under any "
        f"of {PROBLEM_COLUMNS}; inspect the README lines above and add the correct column to "
        "PROBLEM_COLUMNS (or point the live file at the right MATH source). Not guessing a source.")


def _live_rows(cfg: dict[str, Any], spec: dict[str, Any], live_ids: list[str]) -> list[dict]:
    import pyarrow.parquet as pq
    cols = cfg["columns"]
    jk, g = cols["join"], cols["generations"]
    gen_path = os.path.join(os.path.expandvars(cfg["data"]["root"]), spec["folder"],
                            "generations.parquet")
    prob_col = _find_problem_column(gen_path)
    if not prob_col:
        _dump_readme_and_raise(cfg, gen_path)

    want = [str(s) for s in live_ids]
    tbl = pq.read_table(gen_path, columns=[jk["sample_id"], g["gold"], prob_col],
                        filters=[(jk["sample_id"], "in", want)])
    sid = [str(s) for s in tbl.column(jk["sample_id"]).to_pylist()]
    gold = tbl.column(g["gold"]).to_pylist()
    prob = tbl.column(prob_col).to_pylist()
    by_sid: dict[str, tuple] = {}
    for i, s in enumerate(sid):                            # keep the first row seen per sample_id
        by_sid.setdefault(s, (prob[i], gold[i]))
    rows = []
    for s in live_ids:                                     # preserve permutation order
        p, gd = by_sid.get(s, (None, None))
        rows.append({"sample_id": s, "problem": p, "gold": gd, "dataset": "math"})
    missing = [r["sample_id"] for r in rows if r["problem"] is None]
    if missing:
        raise RuntimeError(f"missing problem text for live sample_ids {missing[:5]} in {gen_path}")
    return rows


def run(cfg: dict[str, Any]) -> None:
    out_dir = ensure_dir(SPLITS_DIR / "task3")
    seed = int(cfg["splits"]["seed"])
    dev_size = int(cfg["splits"]["dev_size"])
    n_orders = int(cfg["offline"]["n_orders"])
    order_seed = int(cfg["offline"].get("order_seed", 6304))

    math_eval: list[str] = []
    math_spec = None
    for spec in cfg["data"]["models"]:
        dataset = str(spec["dataset"])
        qa, _ = build_question_arrays(cfg, spec)
        split = make_splits(list(qa), seed=seed, dev_size=dev_size)
        path = save_splits(dataset, split)
        load_or_make_orders(dataset, qa, n_orders=n_orders, seed=order_seed)
        print(f"[prepare] {dataset}: {len(split['dev'])} dev / {len(split['eval'])} eval -> {path}; "
              f"orders written.")
        if dataset == "math":
            math_eval = split["eval"]
            math_spec = spec

    if math_spec is None:
        math_spec = _math_spec(cfg)
    live_ids = live_math_50(math_eval)
    rows = _live_rows(cfg, math_spec, live_ids)
    live_path = out_dir / "math_live_50.json"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[prepare] wrote {live_path} ({len(rows)} questions).")


def main() -> None:
    parser = build_arg_parser("Task 3 prepare: splits, per-question orders, live-50.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
