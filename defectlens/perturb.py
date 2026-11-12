"""Robustness perturbations. Geometric ones transform the ground-truth mask identically."""

import cv2
import numpy as np

PERTURBATIONS = {
    "brightness": [0.6, 0.8, 1.2, 1.4],     # multiplicative factor
    "noise": [5, 10, 20],                   # Gaussian sigma (0-255 scale)
    "blur": [3, 5, 7],                      # Gaussian kernel size
    "jpeg": [90, 70, 50],                   # JPEG quality
    "rotation": [-10, -5, 5, 10],           # degrees
}


def apply(image, mask, kind, level, seed=0):
    img = image.astype(np.float32)
    if kind == "brightness":
        out = img * level
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        out = img + rng.normal(0, level, img.shape)
    elif kind == "blur":
        out = cv2.GaussianBlur(img, (int(level), int(level)), 0)
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(level)])
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    elif kind == "rotation":
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), level, 1.0)
        out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8), mask


def apply_batch(images, masks, kind, level, seed=0):
    pairs = [apply(im, m, kind, level, seed + i) for i, (im, m) in enumerate(zip(images, masks))]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
