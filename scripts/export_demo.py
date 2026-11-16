"""Build the lightweight artifacts the Streamlit demo needs (for Hugging Face Spaces).

Per category:
    - PatchCore memory bank at the demo coreset ratio (default 1%), with a precomputed crop for
      every bank entry so explanations work without the training images
    - thresholds recalibrated for that smaller bank (the score scale changes with the bank)
    - classical model + its thresholds
    - a small gallery of test images with ground-truth masks
Plus results/metrics.md and figures if evaluate.py has been run.

Output: <work_root>/demo_artifacts/   (pushed to the Space by step 7 of notebooks/run_all.ipynb)
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from defectlens.calibration import calibrate, save_thresholds  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_images, load_mask, test_samples, train_calib_split  # noqa: E402
from defectlens.explain import bank_crops  # noqa: E402
from defectlens.patchcore import PatchCore  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    pc_cfg, demo, size, cal = cfg["patchcore"], cfg["demo"], cfg["image_size"], cfg["calibration"]
    work = Path(cfg["work_root"])
    out = work_dir(cfg, "demo_artifacts")

    for cat in cfg["categories"]:
        cat_dir = out / cat
        gal_dir = cat_dir / "gallery"
        gal_dir.mkdir(parents=True, exist_ok=True)
        train, calib = train_calib_split(cfg["data_root"], cat, cfg["calib_fraction"], cfg["seed"])

        if args.force or not (cat_dir / "patchcore.pt").exists():
            print(f"[demo] {cat}: building {demo['coreset_ratio']:.0%} PatchCore bank")
            model = PatchCore(pretrained=pc_cfg["pretrained"], image_size=size,
                              blur_sigma=pc_cfg["blur_sigma"])
            model.fit(load_images(train, size), demo["coreset_ratio"], pc_cfg["projection_dim"],
                      pc_cfg["batch_size"], cfg["seed"], train_paths=train)
            model.bank_crops = bank_crops(model, size)
            th = calibrate(model.predict(load_images(calib, size)), cal["image_percentile"],
                           cal["pixel_percentile"], cal["patch_percentile"])
            model.train_paths = []        # crops replace the training images in the demo
            model.save(cat_dir / "patchcore.pt")
            save_thresholds(th, cat_dir / "patchcore_thresholds.json")

        shutil.copy(work / "models" / f"classical_{cat}.npz", cat_dir / "classical.npz")
        shutil.copy(work / "thresholds" / f"classical_{cat}.json",
                    cat_dir / "classical_thresholds.json")

        samples = test_samples(cfg["data_root"], cat)
        good = [s for s in samples if s.label == 0][: demo["gallery_good"]]
        by_type = {}
        for s in samples:
            if s.label == 1:
                by_type.setdefault(s.defect_type, []).append(s)
        defective = []
        while len(defective) < demo["gallery_defect"] and any(by_type.values()):
            for t in list(by_type):
                if by_type[t] and len(defective) < demo["gallery_defect"]:
                    defective.append(by_type[t].pop(0))
        index = []
        for s in good + defective:
            name = f"{s.defect_type}_{s.path.stem}"
            img = cv2.resize(cv2.imread(str(s.path), cv2.IMREAD_COLOR), (size, size),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(gal_dir / f"{name}.png"), img)
            cv2.imwrite(str(gal_dir / f"{name}_mask.png"), load_mask(s.mask_path, size) * 255)
            index.append({"name": name, "defect_type": s.defect_type, "label": s.label})
        (gal_dir / "index.json").write_text(json.dumps(index, indent=2))
        print(f"[demo] {cat}: gallery of {len(index)} images")

    res_src = work / "results"
    if (res_src / "metrics.md").exists():
        res_dst = out / "results"
        res_dst.mkdir(exist_ok=True)
        shutil.copy(res_src / "metrics.md", res_dst / "metrics.md")
        if (res_src / "figures").exists():
            shutil.copytree(res_src / "figures", res_dst / "figures", dirs_exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Demo artifacts in {out}")


if __name__ == "__main__":
    main()
