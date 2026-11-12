"""Threshold calibration using held-out NORMAL images only (never the test set).

tau_image : PASS/FAIL decision  -> percentile of calibration image scores
tau_pixel : defect masks        -> percentile of calibration anomaly-map pixels
tau_patch : reference level for explanations (raw patch scores before upsampling)

A 99th-percentile rule means roughly 1% of normal images (or pixels) would be flagged.
It is a reasonable starting rule, not a claimed optimum.
"""

import json

import numpy as np


def calibrate(calib_out, image_pct=99, pixel_pct=99, patch_pct=99):
    return {
        "tau_image": float(np.percentile(calib_out["image_scores"], image_pct)),
        "tau_pixel": float(np.percentile(calib_out["maps"], pixel_pct)),
        "tau_patch": float(np.percentile(calib_out["patch_scores"], patch_pct)),
        "image_percentile": image_pct,
        "pixel_percentile": pixel_pct,
        "patch_percentile": patch_pct,
        "n_calibration_images": int(len(calib_out["image_scores"])),
    }


def save_thresholds(thresholds, path):
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(path):
    with open(path) as f:
        return json.load(f)
