"""Task 2 data: COCO captions, annotation-derived grounding, natural VQAv2 pairs, and the seeded
curriculum samplers. (handout p7-p9)

Three components (all selected once with seed 6304; IDs + text saved under splits/task2 and reused
so training never re-parses the large COCO/VQA JSON):
  * COCO captions: 3 captions per training image (6,000 images -> 18,000), 1 per dev image.
  * Annotation-derived grounding (generated programmatically from COCO instances):
      presence: "Does this image contain [object]?" (yes/no), pairing an annotated-object image
                with one lacking that category.
      counting: "How many [objects] can you see?" pairing images with different annotated counts
                in 1..4.
  * Natural VQAv2 complementary pairs: keep pairs whose normalized consensus answers differ with
    >=8/10 annotator agreement; target = original consensus answer string; retain all 10 for
    scoring. Per-type counts yes/no:number:other as in configs/task2/data.yaml.

Curriculum sampling (p9): seeded random sampler; each DRAW contributes two examples (a complete
question pair, or two caption replays); no exact batch/quotas; repeats allowed. Stage mixtures
are in the config. The sampler position is part of the checkpoint (resumable).

Text formatting: visual tokens are spliced BEFORE the question inside the user turn (done in
model.build_inputs); here we produce the *text* token ids and the `insert_at` index marking where
the 196 visual tokens go, plus response-only labels (answer + end-of-turn supervised).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from common import SEED
from common.paths import REPO_ROOT, data_dir, ensure_dir
from .scoring import normalize_answer

IGNORE_INDEX = -100

# Private-use sentinel used to locate the start of the user message content when rendering the chat
# template, so we can splice visual tokens exactly before the question text (it is stripped out and
# never tokenized into the prompt).
_SENTINEL = "\ue000"


# =================================================================================================
# Path + JSON helpers
# =================================================================================================

def _ids_dir(cfg: dict[str, Any]) -> str:
    return os.fspath(REPO_ROOT / cfg.get("ids_dir", "splits/task2"))


def _ann_path(cfg: dict[str, Any], key: str) -> str:
    """Resolve an annotation file under DRIVE_ROOT/data from its config filename."""
    fname = cfg["annotations"][key]
    return os.fspath(data_dir() / fname)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def image_file_name(cfg: dict[str, Any], subset: str, image_id: int) -> str:
    """COCO-2014 file name for an image id, e.g. COCO_train2014_000000391895.jpg."""
    return cfg["paths"]["coco_filename_template"].format(subset=subset, image_id=int(image_id))


# =================================================================================================
# Chat formatting (text side; visual splice happens in model.build_inputs)
# =================================================================================================

def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    limit = min(len(a), len(b))
    while n < limit and a[n] == b[n]:
        n += 1
    return n


def format_vlm_example(tokenizer, question: str, answer: str, *, answer_cap: int,
                       question_cap: int = 80, enable_thinking: bool = False) -> dict[str, Any]:
    """Tokenize one user-question / assistant-answer turn, returning text ids + splice point.

    Returns a dict with:
      input_ids:  prefix (system+user header) ++ question ++ assistant answer ++ end-of-turn
      labels:     IGNORE on everything except the answer tokens + end-of-turn
      insert_at:  index into input_ids where the 196 visual tokens are spliced (before the question)
      n_answer_tokens: supervised token count (for the example-weighted loss denominator)

    Content-token caps (p8): the question text is capped to `question_cap` tokens and the answer to
    `answer_cap` (caption 64 / short-answer 16) before templating.
    """
    question = _cap_text(tokenizer, question, question_cap)
    answer = _cap_text(tokenizer, answer, answer_cap)

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL + question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL + question},
         {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking,
    )
    prompt_text = prompt_text if isinstance(prompt_text, str) else str(prompt_text)
    full_text = full_text if isinstance(full_text, str) else str(full_text)

    pre, prompt_post = prompt_text.split(_SENTINEL, 1)   # pre = up to user content; post = question(+gen prompt)
    _, full_post = full_text.split(_SENTINEL, 1)         # full_post = question + assistant answer + eot

    def tok(s: str) -> list[int]:
        return [int(t) for t in tokenizer(s, add_special_tokens=False)["input_ids"]]

    prefix_ids = tok(pre)
    prompt_post_ids = tok(prompt_post)
    full_post_ids = tok(full_post)

    # Answer begins where the full suffix diverges from the prompt-only suffix (same question text,
    # then the assistant turn). Everything before is masked.
    boundary = _common_prefix_len(prompt_post_ids, full_post_ids)

    input_ids = prefix_ids + full_post_ids
    labels = [IGNORE_INDEX] * (len(prefix_ids) + boundary) + full_post_ids[boundary:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "insert_at": len(prefix_ids),
        "n_answer_tokens": len(full_post_ids) - boundary,
    }


def _cap_text(tokenizer, text: str, cap: int) -> str:
    """Truncate `text` to at most `cap` tokens (decoded back to text), per the content caps (p8)."""
    if cap is None or cap <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= cap:
        return text
    return tokenizer.decode(ids[:cap], skip_special_tokens=True)


# =================================================================================================
# Selection / split freezing (write-once; run via the CLI)
# =================================================================================================

def _rng(*tags: int) -> np.random.Generator:
    """Independent seeded NumPy Generator derived from (SEED, *tags) for reproducible selection."""
    return np.random.default_rng(np.random.SeedSequence([int(SEED), *(int(t) for t in tags)]))


def select_coco_captions(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select COCO caption examples + dev images; save text+IDs to splits/task2. (write-once)

    Rule (TA-question #3, recorded): the first three captions per image in ascending caption-id
    order. 6,000 train2014 images x 3 captions = 18,000; 96 val2014 dev images x 1 caption. Dev
    images are image-disjoint from training by construction (train2014 vs val2014).
    """
    splits = cfg["splits"]["coco_captions"]
    train_caps = _load_json(_ann_path(cfg, "captions_train"))
    dev_caps = _load_json(_ann_path(cfg, "captions_val"))

    train_by_image = _captions_by_image(train_caps)
    dev_by_image = _captions_by_image(dev_caps)

    train_imgs = sorted(train_by_image)
    sel_train = _rng(1).permutation(len(train_imgs))[: int(splits["train_images"])]
    train_examples = []
    for idx in sel_train:
        img = train_imgs[int(idx)]
        for cap in train_by_image[img][:3]:            # first 3 by caption id
            train_examples.append({"image_id": img, "caption": cap})

    dev_imgs = sorted(dev_by_image)
    sel_dev = _rng(2).permutation(len(dev_imgs))[: int(splits["dev_images"])]
    dev_examples = [{"image_id": dev_imgs[int(i)], "caption": dev_by_image[dev_imgs[int(i)]][0]}
                    for i in sel_dev]

    _dump_json(os.path.join(_ids_dir(cfg), "captions_train.json"), train_examples)
    _dump_json(os.path.join(_ids_dir(cfg), "captions_dev.json"), dev_examples)
    return {"train_examples": len(train_examples), "dev_examples": len(dev_examples),
            "train_images": len(sel_train), "dev_images": len(sel_dev)}


def _captions_by_image(coco_caps: dict) -> dict[int, list[str]]:
    """Map image_id -> captions ordered by ascending caption (annotation) id."""
    anns = sorted(coco_caps["annotations"], key=lambda a: a["id"])
    out: dict[int, list[str]] = {}
    for a in anns:
        out.setdefault(int(a["image_id"]), []).append(a["caption"].strip())
    return out


def _instances_by_image(coco_inst: dict) -> tuple[dict[int, dict[int, int]], dict[int, str]]:
    """Return (image_id -> {category_id: count}, category_id -> name) from an instances file."""
    cats = {int(c["id"]): c["name"] for c in coco_inst["categories"]}
    per_image: dict[int, dict[int, int]] = {}
    for a in coco_inst["annotations"]:
        if a.get("iscrowd", 0):
            continue
        img = int(a["image_id"])
        cid = int(a["category_id"])
        per_image.setdefault(img, {})
        per_image[img][cid] = per_image[img].get(cid, 0) + 1
    return per_image, cats


def build_annotation_grounding_pools(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate presence/counting pools (train2014) + freeze dev/test pairs (val2014). (write-once)

    Presence pair: an image containing category c paired with one lacking c -> yes/no targets.
    Counting pair: two images with different annotated counts of c in [1, 4]. Train pools are
    reusable (sampled during training); dev (24) / test (64) pairs are frozen and image-disjoint
    (val2014, dev and test share no images), split evenly presence/counting per the config.
    """
    g = cfg.get("grounding", {})
    lo, hi = g.get("counting_range", [1, 4])
    train_inst, cats = _instances_by_image(_load_json(_ann_path(cfg, "instances_train")))
    val_inst, _ = _instances_by_image(_load_json(_ann_path(cfg, "instances_val")))

    presence_pool = _presence_pairs(train_inst, cats, _rng(3), limit=g.get("max_presence_pool", 8000))
    counting_pool = _counting_pairs(train_inst, cats, _rng(4), lo, hi,
                                    limit=g.get("max_counting_pool", 8000))

    # Dev/test from val2014, image-disjoint between the two sets.
    used: set[int] = set()
    dev_pres = _presence_pairs(val_inst, cats, _rng(5), limit=int(g["dev_presence_pairs"]), exclude=used)
    _mark_used(used, dev_pres, "presence")
    dev_cnt = _counting_pairs(val_inst, cats, _rng(6), lo, hi, limit=int(g["dev_counting_pairs"]),
                              exclude=used)
    _mark_used(used, dev_cnt, "counting")
    test_pres = _presence_pairs(val_inst, cats, _rng(7), limit=int(g["test_presence_pairs"]),
                                exclude=used)
    _mark_used(used, test_pres, "presence")
    test_cnt = _counting_pairs(val_inst, cats, _rng(8), lo, hi, limit=int(g["test_counting_pairs"]),
                               exclude=used)

    _dump_json(os.path.join(_ids_dir(cfg), "presence_pool.json"), presence_pool)
    _dump_json(os.path.join(_ids_dir(cfg), "counting_pool.json"), counting_pool)
    _dump_json(os.path.join(_ids_dir(cfg), "grounding_dev.json"), dev_pres + dev_cnt)
    _dump_json(os.path.join(_ids_dir(cfg), "grounding_test.json"), test_pres + test_cnt)
    return {"presence_pool": len(presence_pool), "counting_pool": len(counting_pool),
            "dev_pairs": len(dev_pres) + len(dev_cnt), "test_pairs": len(test_pres) + len(test_cnt)}


def _mark_used(used: set[int], pairs: list[dict], kind: str) -> None:
    for p in pairs:
        if kind == "presence":
            used.add(p["image_has"]); used.add(p["image_lacks"])
        else:
            used.add(p["image_a"]); used.add(p["image_b"])


def _presence_pairs(per_image, cats, rng, *, limit, exclude: Optional[set[int]] = None) -> list[dict]:
    """Form (has, lacks, category) presence pairs; distinct images, optionally excluding `exclude`."""
    exclude = exclude or set()
    images = [i for i in per_image if i not in exclude]
    img_cats = {i: set(per_image[i]) for i in images}
    pairs: list[dict] = []
    seen: set[int] = set(exclude)
    for cid in rng.permutation(sorted(cats)):
        cid = int(cid)
        has = [i for i in images if cid in img_cats[i] and i not in seen]
        lacks = [i for i in images if cid not in img_cats[i] and i not in seen]
        rng.shuffle(has); rng.shuffle(lacks)
        for h, l in zip(has, lacks):
            if h in seen or l in seen:
                continue
            pairs.append({"type": "presence", "object": cats[cid], "category_id": cid,
                          "image_has": int(h), "image_lacks": int(l)})
            seen.add(h); seen.add(l)
            if len(pairs) >= limit:
                return pairs
    return pairs


def _counting_pairs(per_image, cats, rng, lo, hi, *, limit,
                    exclude: Optional[set[int]] = None) -> list[dict]:
    """Form (image_a@count_a, image_b@count_b, category) pairs with different counts in [lo, hi]."""
    exclude = exclude or set()
    pairs: list[dict] = []
    seen: set[int] = set(exclude)
    for cid in rng.permutation(sorted(cats)):
        cid = int(cid)
        by_count: dict[int, list[int]] = {}
        for i, counts in per_image.items():
            c = counts.get(cid, 0)
            if lo <= c <= hi and i not in seen:
                by_count.setdefault(c, []).append(i)
        counts_present = sorted(by_count)
        for a_idx in range(len(counts_present)):
            for b_idx in range(a_idx + 1, len(counts_present)):
                ca, cb = counts_present[a_idx], counts_present[b_idx]
                la, lb = by_count[ca], by_count[cb]
                rng.shuffle(la); rng.shuffle(lb)
                for ia, ib in zip(la, lb):
                    if ia in seen or ib in seen:
                        continue
                    pairs.append({"type": "counting", "object": cats[cid], "category_id": cid,
                                  "image_a": int(ia), "count_a": int(ca),
                                  "image_b": int(ib), "count_b": int(cb)})
                    seen.add(ia); seen.add(ib)
                    if len(pairs) >= limit:
                        return pairs
    return pairs


def select_natural_vqav2(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select natural VQAv2 complementary pairs by the agreement/consensus rule; freeze splits.

    Keep complementary pairs whose two images have DIFFERENT normalized consensus answers, each
    with >= min_annotator_agreement (8/10) support for its consensus. Target = the original
    consensus answer string; all 10 answers retained for scoring. Per-type (yes/no:number:other)
    counts from the config; train from train2014, dev/test from val2014 (image-disjoint).
    """
    v = cfg["splits"]["natural_vqav2"]
    min_agree = int(v.get("min_annotator_agreement", 8))

    train_pairs = _vqa_qualified_pairs(cfg, "train", min_agree)
    val_pairs = _vqa_qualified_pairs(cfg, "val", min_agree)

    train = _take_by_type(train_pairs, v["train_type_counts"], _rng(9))
    used_imgs: set[int] = set()
    dev = _take_by_type(val_pairs, v["dev_type_counts"], _rng(10), exclude_images=used_imgs)
    for p in dev:
        used_imgs.update(q["image_id"] for q in p["questions"])
    test = _take_by_type(val_pairs, v["test_type_counts"], _rng(11), exclude_images=used_imgs)

    _dump_json(os.path.join(_ids_dir(cfg), "natural_vqa_train.json"), train)
    _dump_json(os.path.join(_ids_dir(cfg), "natural_vqa_dev.json"), dev)
    _dump_json(os.path.join(_ids_dir(cfg), "natural_vqa_test.json"), test)
    return {"train": len(train), "dev": len(dev), "test": len(test)}


def _vqa_qualified_pairs(cfg: dict[str, Any], subset: str, min_agree: int) -> list[dict]:
    """Return qualifying complementary pairs for a VQAv2 subset (train|val), as pair records."""
    q_key = f"vqa_questions_{subset}"
    a_key = f"vqa_annotations_{subset}"
    c_key = f"vqa_complementary_{subset}"
    questions = {int(q["question_id"]): q for q in _load_json(_ann_path(cfg, q_key))["questions"]}
    annotations = {int(a["question_id"]): a for a in _load_json(_ann_path(cfg, a_key))["annotations"]}
    comp_pairs = _load_json(_ann_path(cfg, c_key))

    out: list[dict] = []
    for qid1, qid2 in comp_pairs:
        qid1, qid2 = int(qid1), int(qid2)
        a1, a2 = annotations.get(qid1), annotations.get(qid2)
        if a1 is None or a2 is None:
            continue
        cons1, agree1 = _consensus(a1, min_agree)
        cons2, agree2 = _consensus(a2, min_agree)
        if not (agree1 and agree2):
            continue
        if normalize_answer(cons1) == normalize_answer(cons2):
            continue
        qtype = a1.get("answer_type", "other")
        out.append({
            "pair_id": f"{subset}:{qid1}:{qid2}",
            "answer_type": qtype,
            "questions": [_vqa_q_record(questions[qid1], a1, cons1),
                          _vqa_q_record(questions[qid2], a2, cons2)],
        })
    return out


def _consensus(ann: dict, min_agree: int) -> tuple[str, bool]:
    """Return (consensus answer string, meets-agreement) for one VQA annotation.

    Consensus = the provided multiple_choice_answer; agreement = how many of the 10 raw answers
    normalize to the same string (>= min_agree qualifies).
    """
    consensus = ann.get("multiple_choice_answer", "")
    target_norm = normalize_answer(consensus)
    agree = sum(1 for a in ann["answers"] if normalize_answer(a["answer"]) == target_norm)
    return consensus, agree >= min_agree


def _vqa_q_record(q: dict, ann: dict, consensus: str) -> dict:
    return {
        "question_id": int(q["question_id"]),
        "image_id": int(q["image_id"]),
        "question": q["question"],
        "target": consensus,
        "answers": [a["answer"] for a in ann["answers"]],   # all 10 retained for scoring
        "question_type": ann.get("answer_type", "other"),
    }


def _take_by_type(pairs: list[dict], type_counts: dict[str, int], rng,
                  *, exclude_images: Optional[set[int]] = None) -> list[dict]:
    """Sample `type_counts` qualifying pairs per answer type, avoiding `exclude_images`."""
    exclude_images = exclude_images or set()
    # VQA answer_type values are "yes/no", "number", "other"; config keys use yes_no/number/other.
    key_map = {"yes/no": "yes_no", "number": "number", "other": "other"}
    buckets: dict[str, list[dict]] = {"yes_no": [], "number": [], "other": []}
    for p in pairs:
        k = key_map.get(p["answer_type"], "other")
        if any(q["image_id"] in exclude_images for q in p["questions"]):
            continue
        buckets[k].append(p)

    chosen: list[dict] = []
    taken_imgs: set[int] = set(exclude_images)
    for k, want in type_counts.items():
        pool = buckets.get(k, [])
        for idx in rng.permutation(len(pool)):
            p = pool[int(idx)]
            imgs = [q["image_id"] for q in p["questions"]]
            if any(i in taken_imgs for i in imgs):
                continue
            chosen.append(p)
            taken_imgs.update(imgs)
            if sum(1 for c in chosen if key_map.get(c["answer_type"], "other") == k) >= int(want):
                break
    return chosen


def write_image_manifest(cfg: dict[str, Any]) -> dict[str, int]:
    """Aggregate every selected image id -> subset into the manifest features.py caches from."""
    ids_dir = _ids_dir(cfg)
    manifest: dict[str, str] = {}

    def add(image_id: int, subset: str) -> None:
        manifest[str(int(image_id))] = subset

    for ex in _maybe(os.path.join(ids_dir, "captions_train.json")):
        add(ex["image_id"], "train")
    for ex in _maybe(os.path.join(ids_dir, "captions_dev.json")):
        add(ex["image_id"], "dev")
    for p in _maybe(os.path.join(ids_dir, "presence_pool.json")):
        add(p["image_has"], "train"); add(p["image_lacks"], "train")
    for p in _maybe(os.path.join(ids_dir, "counting_pool.json")):
        add(p["image_a"], "train"); add(p["image_b"], "train")
    for name, subset in [("grounding_dev.json", "dev"), ("grounding_test.json", "test")]:
        for p in _maybe(os.path.join(ids_dir, name)):
            for img in _pair_images(p):
                add(img, subset)
    for name, subset in [("natural_vqa_train.json", "train"), ("natural_vqa_dev.json", "dev"),
                         ("natural_vqa_test.json", "test")]:
        for p in _maybe(os.path.join(ids_dir, name)):
            for q in p["questions"]:
                add(q["image_id"], subset)

    manifest_path = os.fspath(REPO_ROOT / cfg.get("manifest", "splits/task2/image_manifest.json"))
    _dump_json(manifest_path, manifest)
    return {"images": len(manifest)}


def _pair_images(pair: dict) -> list[int]:
    if pair["type"] == "presence":
        return [pair["image_has"], pair["image_lacks"]]
    return [pair["image_a"], pair["image_b"]]


def _maybe(path: str) -> list:
    return _load_json(path) if os.path.exists(path) else []


# =================================================================================================
# Examples + curriculum sampler
# =================================================================================================

@dataclass
class VLMExample:
    """One training example: cached visual features + chat-formatted text + supervised labels."""

    image_id: int
    component: str                      # caption | presence | counting | natural_vqa (+ _replay)
    input_ids: list[int]
    labels: list[int]                   # mask visual tokens, prompts, padding; supervise answer + EOT
    insert_at: int                      # index where the 196 visual tokens splice in
    n_answer_tokens: int
    target: str = ""                    # reference answer (for eval)
    answers: list[str] = field(default_factory=list)   # all human answers (natural VQA scoring)
    question_type: Optional[str] = None
    pair_id: Any = None                 # set for question pairs (kept together for scoring/bootstrap)


class CurriculumSampler:
    """Seeded sampler producing draws per the stage mixture; position is checkpointable.

    A draw first picks a component by the stage mixture, then materializes TWO examples: a complete
    question pair (both images of a presence/counting/natural-VQA pair) or two caption replays. This
    keeps sampling probabilities comparable across components while preserving pairs (p9). Repeats
    are allowed; evaluation images never enter the pools (only train/dev-frozen pools are loaded).
    """

    def __init__(self, cfg: dict[str, Any], stage_cfg: dict[str, Any], tokenizer,
                 *, seed: int = SEED) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.enable_thinking = bool(cfg.get("models", {}).get("language_model", {})
                                    .get("enable_thinking", False))
        caps = cfg.get("train", {}).get("content_token_caps", {})
        self.question_cap = int(caps.get("question", 80))
        self.caption_cap = int(caps.get("caption_target", 64))
        self.answer_cap = int(caps.get("short_answer_target", 16))
        self.prompts = cfg.get("prompts", {})

        self.mixture = dict(stage_cfg["mixture"])
        self._components = list(self.mixture.keys())
        self._probs = np.array([self.mixture[c] for c in self._components], dtype=np.float64)
        self._probs = self._probs / self._probs.sum()

        self._pools = self._load_pools()
        self.rng = np.random.default_rng(seed)
        self.position = 0

    def _ids_path(self, name: str) -> str:
        return os.path.join(_ids_dir(self.cfg), name)

    def _load_pools(self) -> dict[str, list]:
        """Load only the pools referenced by this stage's mixture (caption/presence/counting/vqa)."""
        pools: dict[str, list] = {}
        need = {c.replace("_replay", "") for c in self._components}
        if "caption" in need:
            pools["caption"] = _maybe(self._ids_path("captions_train.json"))
        if "presence" in need:
            pools["presence"] = _maybe(self._ids_path("presence_pool.json"))
        if "counting" in need:
            pools["counting"] = _maybe(self._ids_path("counting_pool.json"))
        if "natural_vqa" in need:
            pools["natural_vqa"] = _maybe(self._ids_path("natural_vqa_train.json"))
        return pools

    # -- example materialization ----------------------------------------------------------------

    def _fmt(self, question: str, answer: str, *, caption: bool) -> dict[str, Any]:
        return format_vlm_example(
            self.tokenizer, question, answer,
            answer_cap=self.caption_cap if caption else self.answer_cap,
            question_cap=self.question_cap, enable_thinking=self.enable_thinking,
        )

    def _caption_example(self, item: dict) -> VLMExample:
        f = self._fmt(self.prompts.get("caption_instruction", "Describe this image."),
                      item["caption"], caption=True)
        return self._make(item["image_id"], "caption", f, target=item["caption"])

    def _presence_examples(self, pair: dict) -> list[VLMExample]:
        q = self.prompts.get("presence_template", "Does this image contain {object}?") \
            .format(object=pair["object"])
        a = self._fmt(q, "yes", caption=False)
        b = self._fmt(q, "no", caption=False)
        return [self._make(pair["image_has"], "presence", a, target="yes", pair_id=_pid(pair)),
                self._make(pair["image_lacks"], "presence", b, target="no", pair_id=_pid(pair))]

    def _counting_examples(self, pair: dict) -> list[VLMExample]:
        q = self.prompts.get("counting_template", "How many {object} can you see?") \
            .format(object=pair["object"])
        a = self._fmt(q, str(pair["count_a"]), caption=False)
        b = self._fmt(q, str(pair["count_b"]), caption=False)
        return [self._make(pair["image_a"], "counting", a, target=str(pair["count_a"]),
                           pair_id=_pid(pair)),
                self._make(pair["image_b"], "counting", b, target=str(pair["count_b"]),
                           pair_id=_pid(pair))]

    def _vqa_examples(self, pair: dict) -> list[VLMExample]:
        out = []
        for q in pair["questions"]:
            f = self._fmt(q["question"], q["target"], caption=False)
            out.append(self._make(q["image_id"], "natural_vqa", f, target=q["target"],
                                  answers=q.get("answers", []), question_type=q.get("question_type"),
                                  pair_id=pair["pair_id"]))
        return out

    def _make(self, image_id, component, f, *, target="", answers=None, question_type=None,
              pair_id=None) -> VLMExample:
        return VLMExample(
            image_id=int(image_id), component=component, input_ids=f["input_ids"],
            labels=f["labels"], insert_at=f["insert_at"], n_answer_tokens=f["n_answer_tokens"],
            target=target, answers=answers or [], question_type=question_type, pair_id=pair_id,
        )

    def _draw_component(self, component: str) -> list[VLMExample]:
        base = component.replace("_replay", "")
        pool = self._pools.get(base, [])
        if not pool:
            return []
        item = pool[int(self.rng.integers(len(pool)))]
        if base == "caption":
            # Two caption replays per draw (p9): sample a second caption item too.
            item2 = pool[int(self.rng.integers(len(pool)))]
            return [self._caption_example(item), self._caption_example(item2)]
        if base == "presence":
            return self._presence_examples(item)
        if base == "counting":
            return self._counting_examples(item)
        if base == "natural_vqa":
            return self._vqa_examples(item)
        return []

    def draws(self, n_updates: int) -> Iterator[list[VLMExample]]:
        """Yield `n_updates` draws, each a list of two VLMExamples per the mixture probabilities."""
        for _ in range(n_updates):
            component = self._components[int(self.rng.choice(len(self._components), p=self._probs))]
            examples = self._draw_component(component)
            self.position += 1
            yield examples

    # -- resume ----------------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return RNG bit-state + draw position for checkpointing."""
        return {"rng": self.rng.bit_generator.state, "position": self.position}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore RNG + position on resume."""
        self.rng.bit_generator.state = state["rng"]
        self.position = int(state["position"])


def _pid(pair: dict) -> str:
    """Stable pair id for a grounding pair (for scoring/bootstrap grouping)."""
    if pair["type"] == "presence":
        return f"presence:{pair['category_id']}:{pair['image_has']}:{pair['image_lacks']}"
    return f"counting:{pair['category_id']}:{pair['image_a']}:{pair['image_b']}"


# =================================================================================================
# CLI: freeze all Task-2 selections + image manifest (write-once; run once on Colab).
# =================================================================================================

def freeze_all(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run every selection step and write the image manifest; return a summary."""
    summary = {
        "coco_captions": select_coco_captions(cfg),
        "grounding": build_annotation_grounding_pools(cfg),
        "natural_vqav2": select_natural_vqav2(cfg),
    }
    summary["manifest"] = write_image_manifest(cfg)
    return summary


def main() -> None:
    from common.config import build_arg_parser, load_yaml
    from common.seeding import seed_all

    parser = build_arg_parser("Freeze Task-2 COCO/VQAv2 selections + image manifest.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    seed_all(SEED)
    cfg = load_yaml(args.config)
    summary = freeze_all(cfg)
    print("[task2.data] froze selections:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
