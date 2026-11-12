"""Check that our AU-PRO (defectlens/metrics.py) matches MVTec's reference algorithm.

MVTec distributes its official evaluation code as a download from mvtec.com, not as a
pip package, so this script compares against a published copy of MVTec's reference AU-PRO
code: utils/au_pro_util.py from the 3D-ADS repository (MIT license), whose header states it
is "based on the official MVTec 3D-AD evaluation code" (same authors and PRO definition as
MVTec AD). It is downloaded at a pinned commit when you run this script; it is not vendored.
If you have the official MVTec AD evaluation code, pass its compute_pro/trapezoid module
with --reference instead.

Expected agreement. Our code follows the MVTec AD evaluator (one curve point per distinct
anomaly score). The 3D-AD copy places thresholds at defect-free pixel scores and interpolates
between them, so even with every threshold the two curves differ by less than one FPR step:
|diff| < 1/N, where N = number of defect-free pixels. That is the tolerance used below.

Two checks:
  1. Synthetic maps, reference at every threshold -> |diff| < 1/N.
  2. Your real PatchCore and classical test maps, reference at 10,000 sampled thresholds
     (exhaustive is too slow in its pure-Python loop) -> should agree to ~1e-3.
Writes <work_root>/results/aupro_parity.txt.
"""

import importlib.util
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from defectlens.config import base_parser, config_from_args, work_dir  # noqa: E402
from defectlens.data import load_test_set  # noqa: E402
from defectlens.metrics import au_pro  # noqa: E402

REF_URL = ("https://raw.githubusercontent.com/eliahuhorwitz/3D-ADS/"
           "612eb7b3ef873212461a836861c480eece7fd6e3/utils/au_pro_util.py")


def load_reference(path=None):
    if path is None:
        path = Path("/tmp/au_pro_reference.py")
        src = urllib.request.urlopen(REF_URL, timeout=30).read().decode()
        # scipy >= 1.12 removed the deprecated scipy.ndimage.measurements alias
        src = src.replace("from scipy.ndimage.measurements import label",
                          "from scipy.ndimage import label")
        path.write_text(src)
    spec = importlib.util.spec_from_file_location("au_pro_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(seed):
    rng = np.random.default_rng(seed)
    n, h = 8, 48
    masks = np.zeros((n, h, h), np.uint8)
    for i in range(n // 2):
        for _ in range(rng.integers(1, 3)):
            y, x = rng.integers(0, h - 10, 2)
            masks[i, y:y + rng.integers(3, 10), x:x + rng.integers(3, 10)] = 1
    maps = rng.random((n, h, h)) + masks * rng.uniform(0.1, 0.6)   # overlapping, imperfect
    return maps, masks


def main():
    p = base_parser(__doc__)
    p.add_argument("--reference", default=None, help="Local path to a reference AU-PRO module")
    p.add_argument("--skip_real", action="store_true", help="Only run the synthetic check")
    args = p.parse_args()
    cfg = config_from_args(args)
    ref = load_reference(args.reference)
    limit = cfg["eval"]["pro_fpr_limit"]
    lines = [f"Reference: {args.reference or REF_URL}", ""]

    lines.append("1) Synthetic maps, reference at every threshold (tolerance 1/N)")
    passed = True
    for seed in range(5):
        maps, masks = synthetic(seed)
        n_ok = int((masks == 0).sum())
        r, _ = ref.calculate_au_pro(list(masks), list(maps), limit, num_thresholds=n_ok)
        ours = au_pro(maps, masks, limit)
        ok = abs(ours - r) < 1.0 / n_ok
        passed &= ok
        lines.append(f"   seed {seed}: ours {ours:.7f}  reference {r:.7f}  |diff| {abs(ours - r):.1e}"
                     f"  (1/N = {1.0 / n_ok:.1e}) {'ok' if ok else 'MISMATCH'}")
    lines.append(f"   -> {'PASS' if passed else 'FAIL'}")

    if not args.skip_real:
        lines += ["", "2) Real test maps, reference at 10,000 sampled thresholds (tolerance 1e-3)"]
        worst = 0.0
        for cat in cfg["categories"]:
            _, _, masks, _ = load_test_set(cfg["data_root"], cat, cfg["image_size"])
            for method in ("patchcore", "classical"):
                path = Path(cfg["work_root"]) / "scores" / f"{method}_{cat}.npz"
                if not path.exists():
                    continue
                maps = np.load(path)["test_maps"].astype(np.float64)
                r, _ = ref.calculate_au_pro(list(masks), list(maps), limit, num_thresholds=10000)
                ours = au_pro(maps, masks, limit)
                worst = max(worst, abs(ours - r))
                lines.append(f"   {cat:10s} {method:10s} ours {ours:.5f}  reference {r:.5f}  "
                             f"|diff| {abs(ours - r):.1e}")
        lines.append(f"   -> {'PASS' if worst < 1e-3 else 'CHECK'} (max |diff| {worst:.1e})")

    report = "\n".join(lines)
    print(report)
    (work_dir(cfg, "results") / "aupro_parity.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
