"""MVTec AD loading: train/calibration split, test images and ground-truth masks.

Expected layout (the official MVTec AD layout):
    <data_root>/<category>/train/good/*.png
    <data_root>/<category>/test/<defect_type or good>/*.png
    <data_root>/<category>/ground_truth/<defect_type>/<stem>_mask.png
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _images_in(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_image(path, size):
    """RGB uint8 array of shape (size, size, 3). Grayscale images are expanded to 3 channels."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path, size):
    """Binary uint8 mask (0/1) resized with nearest-neighbour; zeros if no mask exists."""
    if path is None or not Path(path).exists():
        return np.zeros((size, size), np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def train_calib_split(data_root, category, calib_fraction, seed):
    """Deterministically hold out a fraction of train/good images for threshold calibration."""
    files = _images_in(Path(data_root) / category / "train" / "good")
    if not files:
        raise FileNotFoundError(f"No training images for '{category}' under {data_root}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_calib = max(1, int(round(len(files) * calib_fraction)))
    calib = sorted(files[i] for i in order[:n_calib])
    train = sorted(files[i] for i in order[n_calib:])
    return train, calib


@dataclass
class TestSample:
    path: Path
    defect_type: str
    label: int            # 0 = good, 1 = anomalous
    mask_path: Path | None


def test_samples(data_root, category):
    root = Path(data_root) / category
    samples = []
    for sub in sorted((root / "test").iterdir()):
        if not sub.is_dir():
            continue
        for p in _images_in(sub):
            if sub.name == "good":
                samples.append(TestSample(p, "good", 0, None))
            else:
                mask = root / "ground_truth" / sub.name / f"{p.stem}_mask.png"
                samples.append(TestSample(p, sub.name, 1, mask))
    return samples


def load_images(paths, size):
    return np.stack([load_image(p, size) for p in paths])


def load_test_set(data_root, category, size):
    samples = test_samples(data_root, category)
    images = load_images([s.path for s in samples], size)
    masks = np.stack([load_mask(s.mask_path, size) for s in samples])
    labels = np.array([s.label for s in samples])
    return samples, images, masks, labels
