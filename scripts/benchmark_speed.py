"""Like-for-like speed benchmark: both methods on the SAME device and the SAME images.

The ms/img in metrics.csv comes from each method's own runtime (PatchCore on a T4 GPU,
classical on a CPU runtime), which is not a fair comparison. This script times:
    - PatchCore on CPU and classical on CPU (the fair comparison), and
    - PatchCore on GPU as an extra row, if a GPU is available.
Timings cover inference only (feature extraction + scoring), on images already in memory,
after one warm-up image. Writes <work_root>/results/speed.csv.
"""

import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from defectlens.classical import ClassicalDetector  # noqa: E402
from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_test_set  # noqa: E402
from defectlens.patchcore import FeatureExtractor, PatchCore  # noqa: E402


def timed(fn, images):
    fn(images[:1])                                       # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(len(images)):                         # one image at a time, like the demo
        fn(images[i:i + 1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / len(images)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    pc_cfg, size = cfg["patchcore"], cfg["image_size"]
    n = cfg["eval"]["benchmark_images"]
    models = Path(cfg["work_root"]) / "models"
    cpu_desc = f"cpu ({os.cpu_count()} threads, {platform.processor() or platform.machine()})"
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    extractors = {d.type: FeatureExtractor(pretrained=pc_cfg["pretrained"]).to(d) for d in devices}
    rows = []
    for cat in cfg["categories"]:
        _, images, _, _ = load_test_set(cfg["data_root"], cat, size)
        images = images[:n]
        cl = ClassicalDetector.load(models / f"classical_{cat}.npz")
        rows.append({"category": cat, "method": "Classical (Gabor+LBP)", "device": cpu_desc,
                     "ms_per_img": timed(cl.predict, images)})
        for d in devices:
            pc = PatchCore(device=d, image_size=size, blur_sigma=pc_cfg["blur_sigma"],
                           extractor=extractors[d.type]).load(models / f"patchcore_{cat}.pt")
            rows.append({"category": cat, "method": "PatchCore",
                         "device": cpu_desc if d.type == "cpu" else pc.device_name,
                         "ms_per_img": timed(pc.predict, images)})
        for r in rows[-1 - len(devices):]:
            print(f"[speed] {r['category']:10s} {r['method']:22s} {r['device']:40s} "
                  f"{r['ms_per_img']:8.1f} ms/img")
    df = pd.DataFrame(rows)
    out = work_dir(cfg, "results") / "speed.csv"
    df.to_csv(out, index=False)
    print("\nMean over categories:")
    print(df.groupby(["method", "device"], as_index=False)["ms_per_img"].mean().round(1)
          .to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
