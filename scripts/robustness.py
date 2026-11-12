"""Stretch: robustness study. Perturb test images (and masks for rotations), re-score with
both fitted models, and track image AUROC and AU-PRO.

Needs the models from run_patchcore.py and run_classical.py.
Writes <work_root>/results/robustness.csv and one plot per perturbation type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from defectlens import metrics as M  # noqa: E402
from defectlens.classical import ClassicalDetector  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_test_set  # noqa: E402
from defectlens.patchcore import PatchCore  # noqa: E402
from defectlens.perturb import PERTURBATIONS, apply_batch  # noqa: E402
from defectlens.viz import line_plot  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--kinds", nargs="+", default=list(PERTURBATIONS))
    args = p.parse_args()
    cfg = config_from_args(args)
    pc_cfg, size = cfg["patchcore"], cfg["image_size"]
    models = Path(cfg["work_root"]) / "models"
    res_dir, fig_dir = work_dir(cfg, "results"), work_dir(cfg, "results", "figures")
    out_csv = res_dir / "robustness.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() and not args.force else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["category"], r["method"], r["kind"], float(r["level"])) for r in rows}

    for cat in cfg["categories"]:
        pc = PatchCore(pretrained=pc_cfg["pretrained"], image_size=size,
                       blur_sigma=pc_cfg["blur_sigma"]).load(models / f"patchcore_{cat}.pt")
        cl = ClassicalDetector.load(models / f"classical_{cat}.npz")
        _, images, masks, labels = load_test_set(cfg["data_root"], cat, size)
        jobs = [("none", 0.0)] + [(k, float(v)) for k in args.kinds for v in PERTURBATIONS[k]]
        for kind, level in jobs:
            if all((cat, m, kind, level) in seen for m in ("PatchCore", "Classical")):
                continue
            imgs, msk = (images, masks) if kind == "none" else apply_batch(images, masks, kind, level)
            for name, model in (("PatchCore", pc), ("Classical", cl)):
                out = model.predict(imgs)
                rows.append({"category": cat, "method": name, "kind": kind, "level": level,
                             "img_auroc": M.image_auroc(labels, out["image_scores"]),
                             "au_pro": M.au_pro(out["maps"], msk, cfg["eval"]["pro_fpr_limit"])})
                print(rows[-1])
            pd.DataFrame(rows).to_csv(out_csv, index=False)      # checkpoint after every job

    df = pd.DataFrame(rows)
    base = df[df.kind == "none"]
    for kind in args.kinds:
        sub = pd.concat([base.assign(kind=kind), df[df.kind == kind]])
        mean = sub.groupby(["method", "level"], as_index=False).mean(numeric_only=True)
        for metric in ["img_auroc", "au_pro"]:
            line_plot(mean, "level", metric, "method", fig_dir / f"robust_{kind}_{metric}.png",
                      f"{kind}: {metric} (mean over categories; level 0 = clean)")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
