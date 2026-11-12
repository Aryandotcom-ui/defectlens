"""Classical post-processing: anomaly map -> defect regions -> inspection report.

    threshold (tau_pixel) -> morphological opening (remove specks)
    -> morphological closing (fill small holes) -> connected components (min-area filter)
    -> per-defect bounding box, area, centroid, coarse location, peak score
"""

import cv2
import numpy as np

LOCATION_NAMES = [["top-left", "top", "top-right"],
                  ["left", "center", "right"],
                  ["bottom-left", "bottom", "bottom-right"]]


def defect_mask(anomaly_map, tau_pixel, kernel=5):
    binary = (anomaly_map > tau_pixel).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


def find_defects(anomaly_map, tau_pixel, kernel=5, min_area=30):
    """Returns (cleaned mask with small components removed, list of defect dicts)."""
    mask = defect_mask(anomaly_map, tau_pixel, kernel)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    clean = np.zeros_like(mask)
    defects = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        region = labels == i
        clean[region] = 1
        cx, cy = centroids[i]
        defects.append({
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area_px": int(area),
            "area_pct": 100.0 * area / (h * w),
            "centroid": [float(cx), float(cy)],
            "location": LOCATION_NAMES[min(2, int(3 * cy / h))][min(2, int(3 * cx / w))],
            "peak_score": float(anomaly_map[region].max()),
            "region": region,
        })
    defects.sort(key=lambda d: d["peak_score"], reverse=True)
    return clean, defects


def inspection_report(category, method, image_score, tau_image, defects):
    verdict = "FAIL" if image_score > tau_image else "PASS"
    lines = [f"Product: {category}   Method: {method}   "
             f"Verdict: {verdict} (score {image_score:.2f} {'>' if verdict == 'FAIL' else '<='} "
             f"tau {tau_image:.2f})",
             f"Defects found: {len(defects)}"]
    for i, d in enumerate(defects, 1):
        lines.append(f"  #{i}  area {d['area_px']:>5d} px ({d['area_pct']:.2f}%)  "
                     f"location: {d['location']:<12s} peak {d['peak_score']:.2f}")
    if verdict == "FAIL" and not defects:
        lines.append("  (image-level anomaly without a localized region above tau_pixel)")
    if verdict == "PASS" and defects:
        lines.append("  (regions above tau_pixel found, but image score below tau_image)")
    return {"verdict": verdict, "image_score": float(image_score), "tau_image": float(tau_image),
            "n_defects": len(defects), "text": "\n".join(lines)}
