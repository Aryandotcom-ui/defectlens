"""Self-contained checks (no downloads): python tests/test_metrics.py

AU-PRO is compared with a direct, slow implementation of its definition: for every distinct
threshold, FPR over defect-free pixels and the mean overlap over all 8-connected ground-truth
regions, integrated with the trapezoidal rule to the FPR limit (interpolating at the limit).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.ndimage import label  # noqa: E402

from defectlens.metrics import au_pro, pixel_f1_iou  # noqa: E402


def brute_force_au_pro(maps, masks, limit):
    ok = masks == 0
    regions = []
    for i, m in enumerate(masks):
        lab, n = label(m, np.ones((3, 3)))
        regions += [(i, lab == k) for k in range(1, n + 1)]
    fprs, pros = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        pred = maps >= t
        fprs.append(pred[ok].mean())
        pros.append(np.mean([pred[i][r].mean() for i, r in regions]))
    fprs, pros = np.array(fprs + [1.0]), np.array(pros + [1.0])
    sel = fprs <= limit
    area = np.sum(0.5 * (pros[sel][1:] + pros[sel][:-1]) * np.diff(fprs[sel]))
    j = np.argmax(~sel)
    y = pros[j - 1] + (pros[j] - pros[j - 1]) * (limit - fprs[j - 1]) / (fprs[j] - fprs[j - 1])
    area += 0.5 * (pros[j - 1] + y) * (limit - fprs[j - 1])
    return area / limit


def test_au_pro_matches_definition():
    rng = np.random.default_rng(0)
    for _ in range(5):
        masks = np.zeros((6, 20, 20), np.uint8)
        masks[0, 2:6, 2:6] = 1
        masks[1, 10:18, 3:5] = 1
        masks[1, 1:3, 15:19] = 1
        masks[3, 5:15, 5:15] = 1
        maps = rng.random((6, 20, 20)) + masks * rng.uniform(0.2, 0.8)
        assert abs(au_pro(maps, masks, 0.3) - brute_force_au_pro(maps, masks, 0.3)) < 1e-9


def test_perfect_and_iou():
    masks = np.zeros((2, 10, 10), np.uint8)
    masks[0, 2:5, 2:5] = 1
    assert abs(au_pro(masks.astype(float), masks, 0.3) - 1.0) < 1e-9
    m = pixel_f1_iou(masks, masks)
    assert m["px_iou"] == 1.0 and m["px_f1"] == 1.0


if __name__ == "__main__":
    test_au_pro_matches_definition()
    test_perfect_and_iou()
    print("all metric tests passed")
