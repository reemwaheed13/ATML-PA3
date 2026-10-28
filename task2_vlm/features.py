"""SigLIP feature extraction and FP16 caching. (handout p7)

Extract the 196 final patch features in their 14x14 spatial order from
google/siglip-base-patch16-224, applying SigLIP's final layer normalization exactly once. Cache
to disk on DRIVE_ROOT in FP16 with DETERMINISTIC image preprocessing, then remove the vision
encoder from GPU memory before training Qwen. Caching is a one-time cost that must be logged
separately (Required Evidence, p9).

Only images in the frozen split ID files are cached; eval images must not leak into training.
"""

from __future__ import annotations

from typing import Any, Iterable


def build_siglip(cfg: dict[str, Any]):
    """Load the frozen SigLIP encoder + its image processor (deterministic preprocessing).

    TODO: AutoModel / AutoImageProcessor for siglip-base-patch16-224; set eval(); no grad.
    """
    raise NotImplementedError


def extract_features(model, processor, image_paths: Iterable[str]):
    """Yield (image_id, feats) where feats is [196, 768] FP16 after the final layer norm.

    TODO: batch images with deterministic preprocessing; forward; take the 196 patch tokens in
          14x14 order; apply final layer norm once; cast FP16.
    """
    raise NotImplementedError


def cache_features(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract + persist features for all split images to DRIVE_ROOT/task2/features; log cost.

    Returns a resource summary (image count, wall time, bytes) for the resource table.

    TODO: resolve image IDs from splits/task2; skip already-cached; write per-image or sharded
          FP16 tensors; record timing + size; free the encoder afterward.
    """
    raise NotImplementedError


def load_cached_features(image_id: str):
    """Load the cached [196, 768] FP16 feature tensor for one image.

    TODO: read from the feature cache path for image_id.
    """
    raise NotImplementedError


def main() -> None:
    """CLI: cache features for the configured splits. TODO: parse --config; call cache_features."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
