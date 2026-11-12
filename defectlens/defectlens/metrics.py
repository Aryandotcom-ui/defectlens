"""Evaluation metrics.

Threshold-free (primary):
    image AUROC, pixel AUROC, AU-PRO up to 30% FPR.
At calibrated thresholds (secondary):
    image precision / recall / F1, FPR on normal test images, pixel F1 and IoU.

AU-PRO follows the MVTec AD evaluation protocol (Bergmann et al., IJCV 2021 and the official
evaluation scripts): per-region overlap averaged over all ground-truth connected components
(8-connectivity), plotted against the false positive rate on defect-free pixels, integrated
with the trapezoidal rule up to the FPR limit and normalized by that limit.
"""

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def image_auroc(labels, image_scores):
    return float(roc_auc_score(labels, image_scores))


def pixel_auroc(masks, maps):
    m = masks.ravel()
    if m.min() == m.max():
        return float("nan")
    return float(roc_auc_score(m, maps.ravel()))


def pro_curve(maps, masks):
    """FPR and PRO for every distinct threshold (official MVTec algorithm)."""
    structure = np.ones((3, 3), dtype=int)
    fp_changes = np.zeros(maps.shape, dtype=np.uint32)
    pro_changes = np.zeros(maps.shape, dtype=np.float64)
    n_ok, n_regions = 0, 0
    for i, gt in enumerate(masks):
        labeled, n_comp = label(gt, structure)
        n_regions += n_comp
        ok = labeled == 0
        n_ok += int(ok.sum())
        fp_changes[i, ok] = 1
        for k in range(1, n_comp + 1):
            region = labeled == k
            pro_changes[i, region] = 1.0 / region.sum()
    if n_regions == 0 or n_ok == 0:
        return None, None
    scores = maps.ravel()
    order = np.argsort(scores, kind="stable")[::-1]
    scores = scores[order]
    fprs = np.cumsum(fp_changes.ravel()[order]) / n_ok
    pros = np.cumsum(pro_changes.ravel()[order]) / n_regions
    keep = np.append(np.diff(scores) != 0, True)         # one point per distinct threshold
    fprs = np.clip(fprs[keep], 0, 1)
    pros = np.clip(pros[keep], 0, 1)
    return np.concatenate(([0.0], fprs, [1.0])), np.concatenate(([0.0], pros, [1.0]))


def trapezoid_area(x, y, x_max):
    """Area under y(x) from 0 to x_max, interpolating linearly at x_max."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = x <= x_max
    area = np.trapezoid(y[mask], x[mask]) if hasattr(np, "trapezoid") else np.trapz(y[mask], x[mask])
    if not mask.all():
        i = np.argmax(~mask)                   # first point beyond x_max
        x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
        y_at = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0) if x1 > x0 else y0
        area += 0.5 * (y0 + y_at) * (x_max - x0)
    return float(area)


def au_pro(maps, masks, fpr_limit=0.3):
    fprs, pros = pro_curve(maps, masks)
    if fprs is None:
        return float("nan")
    return trapezoid_area(fprs, pros, fpr_limit) / fpr_limit


def thresholded_image_metrics(labels, image_scores, tau_image):
    pred = image_scores > tau_image
    labels = labels.astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"img_precision": precision, "img_recall": recall, "img_f1": f1, "img_fpr": fpr}


def pixel_f1_iou(pred_masks, gt_masks):
    """Computed over all test pixels at once (pred_masks = post-processed defect masks)."""
    p, g = pred_masks.astype(bool), gt_masks.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    iou = tp / (tp + fp + fn) if tp else 0.0
    return {"px_f1": f1, "px_iou": iou}
