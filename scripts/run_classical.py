"""Fit the classical Gabor + LBP baseline per category and score calibration + test sets.

Outputs (resumable):
    <work_root>/models/classical_<category>.npz
    <work_root>/scores/classical_<category>.npz
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from defectlens.classical import ClassicalDetector  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_images, load_test_set, train_calib_split  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    c, size = cfg["classical"], cfg["image_size"]
    models, scores = work_dir(cfg, "models"), work_dir(cfg, "scores")

    for cat in cfg["categories"]:
        model_path = models / f"classical_{cat}.npz"
        score_path = scores / f"classical_{cat}.npz"
        if score_path.exists() and not args.force:
            print(f"[classical] {cat}: cached, skipping")
            continue
        t0 = time.time()
        train, calib = train_calib_split(cfg["data_root"], cat, cfg["calib_fraction"], cfg["seed"])
        if model_path.exists() and not args.force:
            det = ClassicalDetector.load(model_path)
        else:
            print(f"[classical] {cat}: fitting on {len(train)} images")
            det = ClassicalDetector(size, c["gabor_wavelengths"], c["gabor_orientations"],
                                    c["lbp_points"], c["lbp_radius"], c["blur_sigma"])
            det.fit(load_images(train, size))
            det.save(model_path)

        calib_out = det.predict(load_images(calib, size))
        samples, images, _, labels = load_test_set(cfg["data_root"], cat, size)
        test_out = det.predict(images)
        np.savez_compressed(
            score_path,
            calib_image_scores=calib_out["image_scores"],
            calib_maps=calib_out["maps"].astype(np.float16),
            calib_patch_scores=calib_out["patch_scores"],
            test_image_scores=test_out["image_scores"],
            test_maps=test_out["maps"].astype(np.float16),
            test_patch_scores=test_out["patch_scores"],
            test_labels=labels,
            test_paths=np.array([str(s.path) for s in samples]),
            seconds_per_image=test_out["seconds_per_image"],
            device="cpu",
        )
        print(f"[classical] {cat}: {test_out['seconds_per_image'] * 1000:.0f} ms/img, "
              f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
