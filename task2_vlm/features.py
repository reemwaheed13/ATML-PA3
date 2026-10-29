"""SigLIP feature extraction and FP16 caching. (handout p7)

Extract the 196 final patch features in their 14x14 spatial order from
google/siglip-base-patch16-224, applying SigLIP's final layer normalization exactly once. Cache
to disk on DRIVE_ROOT in FP16 with DETERMINISTIC image preprocessing, then remove the vision
encoder from GPU memory before training Qwen. Caching is a one-time cost that must be logged
separately (Required Evidence, p9).

Only images in the frozen split manifest are cached; eval images never leak into training because
the manifest is built from the frozen split IDs (train/dev/test are image-disjoint by construction).

SigLIP's vision transformer applies `post_layernorm` to `last_hidden_state` internally, so taking
`last_hidden_state` gives the 196 patch tokens with the final layer norm applied exactly once (no
CLS token; 224/16 = 14, 14x14 = 196, row-major spatial order).
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Iterator

import torch

from common.paths import data_dir, ensure_dir, task2_dir
from .data import image_file_name


def build_siglip(cfg: dict[str, Any]):
    """Load the frozen SigLIP vision tower + its image processor (deterministic preprocessing)."""
    from transformers import AutoImageProcessor
    try:
        from transformers import SiglipVisionModel
        name = cfg["vision_encoder"]["name"]
        model = SiglipVisionModel.from_pretrained(name)
    except Exception:
        # Fallback: load the full SigLIP model and take its vision tower.
        from transformers import AutoModel
        name = cfg["vision_encoder"]["name"]
        model = AutoModel.from_pretrained(name).vision_model

    processor = AutoImageProcessor.from_pretrained(name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


@torch.no_grad()
def extract_features(model, processor, image_paths: Iterable[str], *, device=None,
                     batch_size: int = 32) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (image_path, feats) where feats is [196, 768] FP16 after the final layer norm.

    Images are preprocessed deterministically by the SigLIP processor (fixed resize to 224, no
    random crop), batched for throughput; the encoder runs in its loaded (FP32) precision and the
    output is cast to FP16 for caching.
    """
    from PIL import Image

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    paths = list(image_paths)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        out = model(pixel_values=pixel_values)
        feats = out.last_hidden_state                 # [B, 196, 768], post-layernorm
        feats = feats.to(torch.float16).cpu()
        for p, f in zip(batch_paths, feats):
            yield p, f.clone()


def _image_path(cfg: dict[str, Any], subset_key: str, image_id: int) -> str:
    """Resolve the on-disk path of a COCO image given its manifest subset key (train|dev|test)."""
    subset = cfg["paths"]["images"][subset_key]            # e.g. train2014 / val2014
    return os.fspath(data_dir() / subset / image_file_name(cfg, subset, image_id))


def feature_cache_dir(cfg: dict[str, Any]):
    return ensure_dir(task2_dir(cfg.get("features", {}).get("cache_subdir", "features")))


def _cache_path(cfg: dict[str, Any], image_id: int) -> str:
    return os.fspath(feature_cache_dir(cfg) / f"{int(image_id)}.pt")


def cache_features(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract + persist features for all manifest images to DRIVE_ROOT/task2/features; log cost.

    Skips images already cached (idempotent/resumable). Returns a resource summary (image count,
    wall time, bytes) for the Task-2 resource table; the vision encoder is freed before returning.
    """
    from common.paths import REPO_ROOT
    import json

    manifest_path = os.fspath(REPO_ROOT / cfg.get("manifest", "splits/task2/image_manifest.json"))
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest: dict[str, str] = json.load(f)

    cache = feature_cache_dir(cfg)
    todo: list[tuple[str, str]] = []            # (cache_path, image_path)
    for image_id, subset_key in manifest.items():
        cp = _cache_path(cfg, int(image_id))
        if not os.path.exists(cp):
            todo.append((cp, _image_path(cfg, subset_key, int(image_id))))

    model, processor = build_siglip(cfg)
    bs = int(cfg.get("features", {}).get("batch_size", 32))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    cached = 0
    cache_paths = [cp for cp, _ in todo]
    for (cp, _), (_, feats) in zip(todo, extract_features(
            model, processor, [ip for _, ip in todo], device=device, batch_size=bs)):
        torch.save(feats, cp)
        cached += 1
    wall = time.perf_counter() - t0

    # Free the encoder from GPU before Qwen training (p7).
    del model
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        peak = 0

    total = len(manifest)
    total_bytes = sum(os.path.getsize(_cache_path(cfg, int(i))) for i in manifest
                      if os.path.exists(_cache_path(cfg, int(i))))
    return {
        "type": "feature_cache",
        "images_total": total,
        "images_cached_now": cached,
        "images_already_cached": total - len(todo),
        "wall_seconds": wall,
        "peak_gpu_bytes": peak,
        "cache_bytes": int(total_bytes),
    }


def load_cached_features(cfg: dict[str, Any], image_id: int, *, device=None,
                         dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Load the cached [196, 768] feature tensor for one image (optionally moved to device/dtype)."""
    feats = torch.load(_cache_path(cfg, int(image_id)), map_location="cpu")
    if device is not None:
        feats = feats.to(device)
    return feats.to(dtype)


def main() -> None:
    from common.config import build_arg_parser, load_yaml
    from common.logging_utils import JsonlLogger
    from common.paths import task2_dir

    parser = build_arg_parser("Cache SigLIP features for the frozen Task-2 image manifest.")
    args = parser.parse_args()
    if args.drive_root:
        os.environ["DRIVE_ROOT"] = args.drive_root
    cfg = load_yaml(args.config)

    summary = cache_features(cfg)
    results = ensure_dir(task2_dir("results") / "features")
    logger = JsonlLogger(os.fspath(results / "feature_cache.jsonl"), append=True)
    logger.log(summary)
    logger.close()
    print(f"[task2.features] cached {summary['images_cached_now']} new "
          f"({summary['images_already_cached']} already); {summary['wall_seconds']:.1f}s, "
          f"peak GPU {summary['peak_gpu_bytes']} bytes, cache {summary['cache_bytes']} bytes.")


if __name__ == "__main__":
    main()
