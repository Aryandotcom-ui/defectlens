"""Calibrate thresholds, compute all metrics and make figures for both methods.

Reads <work_root>/scores/*.npz (from run_patchcore.py / run_classical.py) and writes:
    <work_root>/thresholds/<method>_<category>.json
    <work_root>/results/metrics.csv, metrics.md
    <work_root>/results/figures/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from defectlens import metrics as M  # noqa: E402
from defectlens.calibration import calibrate, save_thresholds  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_test_set  # noqa: E402
from defectlens.postprocess import find_defects  # noqa: E402
from defectlens.viz import heatmap_overlay, metric_bars, qualitative_grid  # noqa: E402

METHODS = {"patchcore": "PatchCore", "classical": "Classical (Gabor+LBP)"}


def load_scores(path):
    z = np.load(path, allow_pickle=False)
    calib = {"image_scores": z["calib_image_scores"], "maps": z["calib_maps"].astype(np.float32),
             "patch_scores": z["calib_patch_scores"]}
    test = {k[5:]: z[k] for k in z.files if k.startswith("test_")}
    test["maps"] = test["maps"].astype(np.float32)
    extra = {k: z[k] for k in z.files if not k.startswith(("calib_", "test_"))}
    return calib, test, extra


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    cal, post, size = cfg["calibration"], cfg["postprocess"], cfg["image_size"]
    thr_dir, res_dir = work_dir(cfg, "thresholds"), work_dir(cfg, "results")
    fig_dir = work_dir(cfg, "results", "figures")
    rows = []

    for cat in cfg["categories"]:
        samples, images, masks, labels = load_test_set(cfg["data_root"], cat, size)
        overlays = {}
        for method, label in METHODS.items():
            path = Path(cfg["work_root"]) / "scores" / f"{method}_{cat}.npz"
            if not path.exists():
                print(f"[eval] missing {path.name}, skipping")
                continue
            calib, test, extra = load_scores(path)
            assert len(test["image_scores"]) == len(labels), "test set changed since scoring"
            th = calibrate(calib, cal["image_percentile"], cal["pixel_percentile"],
                           cal["patch_percentile"])
            save_thresholds(th, thr_dir / f"{method}_{cat}.json")

            pred_masks = np.stack([find_defects(m, th["tau_pixel"], post["kernel"],
                                                post["min_area"])[0] for m in test["maps"]])
            row = {
                "category": cat, "method": label,
                "img_auroc": M.image_auroc(labels, test["image_scores"]),
                "px_auroc": M.pixel_auroc(masks, test["maps"]),
                "au_pro": M.au_pro(test["maps"], masks, cfg["eval"]["pro_fpr_limit"]),
                **M.thresholded_image_metrics(labels, test["image_scores"], th["tau_image"]),
                **M.pixel_f1_iou(pred_masks, masks),
                "ms_per_img": float(extra["seconds_per_image"]) * 1000,
                "timing_device": str(extra["device"]) if "device" in extra else "unknown",
            }
            if "bank_mb" in extra:
                row["bank_mb"] = float(extra["bank_mb"])
            rows.append(row)
            print(f"[eval] {cat:10s} {label:22s} img-AUROC {row['img_auroc']:.3f}  "
                  f"px-AUROC {row['px_auroc']:.3f}  AU-PRO {row['au_pro']:.3f}  "
                  f"img-F1 {row['img_f1']:.3f}  px-IoU {row['px_iou']:.3f}")

            vmin = float(np.median(calib["maps"]))
            vmax = float(np.percentile(test["maps"], 99.9))
            overlays[method] = [heatmap_overlay(images[i], test["maps"][i], vmin, vmax)
                                for i in range(len(images))]

        if len(overlays) == 2:
            anomalous = [i for i in range(len(labels)) if labels[i] == 1]
            pick = anomalous[:: max(1, len(anomalous) // 4)][:4]
            good = [i for i in range(len(labels)) if labels[i] == 0][:1]
            grid_rows = [{"image": images[i], "gt": masks[i] * 255,
                          "pc_overlay": overlays["patchcore"][i],
                          "cl_overlay": overlays["classical"][i],
                          "label": samples[i].defect_type} for i in pick + good]
            qualitative_grid(grid_rows, fig_dir / f"qualitative_{cat}.png", title=cat)

    if not rows:
        print("No scores found. Run run_patchcore.py / run_classical.py first.")
        return
    df = pd.DataFrame(rows)
    mean = df.groupby("method", as_index=False).mean(numeric_only=True)
    mean.insert(0, "category", "MEAN")
    mean["timing_device"] = mean["method"].map(
        df.groupby("method")["timing_device"].agg(lambda d: ", ".join(sorted(set(d)))))
    full = pd.concat([df, mean], ignore_index=True)
    full.to_csv(res_dir / "metrics.csv", index=False)
    (res_dir / "metrics.md").write_text(full.round(3).to_markdown(index=False))
    for metric in ["img_auroc", "px_auroc", "au_pro", "img_f1", "px_iou"]:
        metric_bars(df, metric, fig_dir / f"bars_{metric}.png")
    print("\nNote: ms_per_img was measured on each method's own runtime (see timing_device), so it "
          "is not a like-for-like speed comparison. Use scripts/benchmark_speed.py for that.")
    print(f"\nWrote {res_dir / 'metrics.md'}\n")
    print(full.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
