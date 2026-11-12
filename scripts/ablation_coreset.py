"""Stretch: coreset-ratio ablation for PatchCore (accuracy vs. memory vs. speed).

Training features come from the cache written by run_patchcore.py (<work_root>/features/),
so nothing is re-extracted when that cache exists. The memory bank is rebuilt at each ratio.
Progress is checkpointed to <work_root>/results/ablation_coreset.csv after every
(category, ratio), so a disconnected run resumes where it stopped.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from defectlens import metrics as M  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_images, load_test_set, train_calib_split  # noqa: E402
from defectlens.patchcore import PatchCore, extract  # noqa: E402
from defectlens.viz import line_plot  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.10, 0.25])
    args = p.parse_args()
    cfg = config_from_args(args)
    pc_cfg, size = cfg["patchcore"], cfg["image_size"]
    res_dir = work_dir(cfg, "results")
    fig_dir = work_dir(cfg, "results", "figures")
    feat_dir = work_dir(cfg, "features")
    out_csv = res_dir / "ablation_coreset.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() and not args.force else []
    done = {(r["category"], float(r["coreset_ratio"])) for r in rows}

    for cat in cfg["categories"]:
        todo = [r for r in args.ratios if (cat, float(r)) not in done]
        if not todo:
            print(f"[ablation] {cat}: done, skipping")
            continue
        train, _ = train_calib_split(cfg["data_root"], cat, cfg["calib_fraction"], cfg["seed"])
        model = PatchCore(pretrained=pc_cfg["pretrained"], image_size=size,
                          blur_sigma=pc_cfg["blur_sigma"])
        train_feats = model.extract_cached(load_images(train, size), feat_dir / f"train_{cat}.pt",
                                           pc_cfg["batch_size"])
        _, images, masks, labels = load_test_set(cfg["data_root"], cat, size)
        t0 = time.perf_counter()
        test_feats = extract(model.extractor, images, model.device, pc_cfg["batch_size"])
        extract_s = (time.perf_counter() - t0) / len(images)
        for r in todo:
            model.fit_from_features(train_feats, r, pc_cfg["projection_dim"], cfg["seed"], train)
            t0 = time.perf_counter()
            out = model.score_features(test_feats)
            knn_s = (time.perf_counter() - t0) / len(images)
            rows.append({"category": cat, "coreset_ratio": r, "bank_size": len(model.bank),
                         "bank_mb": model.bank_megabytes,
                         "img_auroc": M.image_auroc(labels, out["image_scores"]),
                         "au_pro": M.au_pro(out["maps"], masks, cfg["eval"]["pro_fpr_limit"]),
                         "ms_per_img": 1000 * (extract_s + knn_s),
                         "timing_device": model.device_name})
            print(rows[-1])
            pd.DataFrame(rows).to_csv(out_csv, index=False)      # checkpoint
    df = pd.DataFrame(rows)
    mean = df.groupby("coreset_ratio", as_index=False).mean(numeric_only=True)
    mean["series"] = "mean over categories"
    for metric in ["img_auroc", "au_pro", "ms_per_img"]:
        line_plot(mean, "coreset_ratio", metric, "series",
                  fig_dir / f"ablation_{metric}.png", f"Coreset ratio vs {metric}")
    print(mean.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
