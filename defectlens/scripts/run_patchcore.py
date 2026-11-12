"""Fit PatchCore per category and score the calibration + test sets.

Outputs (resumable: finished categories are skipped unless --force):
    <work_root>/features/train_<category>.pt   (training features before coreset, float16)
    <work_root>/models/patchcore_<category>.pt
    <work_root>/scores/patchcore_<category>.npz
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_images, load_test_set, train_calib_split  # noqa: E402
from defectlens.patchcore import PatchCore  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    pc_cfg, size = cfg["patchcore"], cfg["image_size"]
    models, scores = work_dir(cfg, "models"), work_dir(cfg, "scores")
    features = work_dir(cfg, "features") if pc_cfg.get("cache_features", True) else None

    for cat in cfg["categories"]:
        model_path = models / f"patchcore_{cat}.pt"
        score_path = scores / f"patchcore_{cat}.npz"
        if score_path.exists() and not args.force:
            print(f"[patchcore] {cat}: cached, skipping")
            continue
        t0 = time.time()
        train, calib = train_calib_split(cfg["data_root"], cat, cfg["calib_fraction"], cfg["seed"])
        model = PatchCore(pretrained=pc_cfg["pretrained"], image_size=size,
                          blur_sigma=pc_cfg["blur_sigma"])
        if model_path.exists() and not args.force:
            model.load(model_path)
        else:
            print(f"[patchcore] {cat}: fitting on {len(train)} images ({model.device_name})")
            cache = features / f"train_{cat}.pt" if features else None
            if args.force and cache is not None and cache.exists():
                cache.unlink()
            feats = model.extract_cached(load_images(train, size), cache, pc_cfg["batch_size"])
            model.fit_from_features(feats, pc_cfg["coreset_ratio"], pc_cfg["projection_dim"],
                                    cfg["seed"], train_paths=train)
            model.save(model_path)
            del feats

        calib_out = model.predict(load_images(calib, size), pc_cfg["batch_size"])
        samples, images, _, labels = load_test_set(cfg["data_root"], cat, size)
        test_out = model.predict(images, pc_cfg["batch_size"])
        np.savez_compressed(
            score_path,
            calib_image_scores=calib_out["image_scores"],
            calib_maps=calib_out["maps"].astype(np.float16),
            calib_patch_scores=calib_out["patch_scores"],
            test_image_scores=test_out["image_scores"],
            test_maps=test_out["maps"].astype(np.float16),
            test_patch_scores=test_out["patch_scores"],
            test_nn_idx=test_out["nn_idx"],
            test_labels=labels,
            test_paths=np.array([str(s.path) for s in samples]),
            seconds_per_image=test_out["seconds_per_image"],
            device=model.device_name,
            bank_size=len(model.bank),
            bank_mb=model.bank_megabytes,
        )
        print(f"[patchcore] {cat}: bank {len(model.bank)} patches ({model.bank_megabytes:.1f} MB), "
              f"{test_out['seconds_per_image'] * 1000:.0f} ms/img on {model.device_name}, done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
