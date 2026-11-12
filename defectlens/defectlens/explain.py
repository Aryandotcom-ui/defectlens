"""Retrieval-based explanation: show the nearest NORMAL patch each flagged region was compared to.

For each detected defect region (PatchCore):
    1. take the highest-scoring grid cell inside the region,
    2. look up its nearest memory-bank entry -> (train_image_idx, row, col),
    3. return the crop around the flagged cell and the crop around the matched normal cell,
       with the feature distance relative to the calibration reference (tau_patch).

Caveats (shown in the UI and report):
    - This is evidence ("nothing in the normal data looks like this; here is the closest"),
      not a causal explanation of the defect.
    - Spatial precision is approximate: a grid cell covers 8x8 pixels but its receptive field
      is much larger, so crops indicate the region a feature describes, not exact pixels.
"""

import numpy as np

from .data import load_image

CROP = 40


def crop_at(image, row, col, stride, size=CROP):
    """Crop of `size` pixels centred on grid cell (row, col), shifted to stay inside the image."""
    h, w = image.shape[:2]
    cy, cx = row * stride + stride // 2, col * stride + stride // 2
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return image[y0:y0 + size, x0:x0 + size].copy(), (x0, y0, size, size)


def normal_crop(model, bank_index, stride, image_size):
    if model.bank_crops is not None:
        return model.bank_crops[bank_index]
    img_idx, row, col = model.bank_meta[bank_index]
    image = load_image(model.train_paths[img_idx], image_size)
    return crop_at(image, row, col, stride)[0]


def explain_defects(model, image, patch_scores, nn_idx, defects, tau_patch, max_items=3):
    """patch_scores, nn_idx: (g, g) arrays for this image from PatchCore.predict."""
    g = patch_scores.shape[0]
    size = image.shape[0]
    stride = size // g
    evidence = []
    for d in defects[:max_items]:
        cells = d["region"].reshape(g, stride, g, stride).any(axis=(1, 3))
        if not cells.any():
            continue
        masked = np.where(cells, patch_scores, -np.inf)
        row, col = np.unravel_index(np.argmax(masked), masked.shape)
        bank_index = int(nn_idx[row, col])
        flagged, box = crop_at(image, row, col, stride)
        dist = float(patch_scores[row, col])
        evidence.append({
            "cell": (int(row), int(col)),
            "flagged_crop": flagged,
            "flagged_box": box,
            "normal_crop": normal_crop(model, bank_index, stride, size),
            "distance": dist,
            "tau_patch": float(tau_patch),
            "ratio": dist / tau_patch if tau_patch > 0 else float("nan"),
        })
    return evidence


def bank_crops(model, image_size):
    """Precompute a crop for every memory-bank entry (lets the demo run without train images)."""
    stride = image_size // model.grid
    cache, crops = {}, []
    for img_idx, row, col in model.bank_meta:
        if img_idx not in cache:
            cache[img_idx] = load_image(model.train_paths[img_idx], image_size)
        crops.append(crop_at(cache[img_idx], row, col, stride)[0])
    return np.stack(crops).astype(np.uint8)
