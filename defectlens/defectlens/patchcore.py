"""A custom PatchCore-style anomaly detector, implemented from scratch.

Based on Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
It follows the paper's core design (pretrained mid-level features, local neighbourhood
aggregation, greedy coreset memory bank, nearest-neighbour scoring) but is NOT a bit-for-bit
reproduction of the official implementation. Deliberate differences:
    - Preprocessing: direct resize to 224x224. Official: resize 256 -> center crop 224
      (which can cut off border defects and requires cropping the masks identically).
    - Embedding: plain 3x3 average pooling + concatenation -> 1536 dims. Official: 3x3 patch
      unfolding, per-layer adaptive pooling to 1024 dims, then aggregation to 1024 dims.
    - Coreset: exact k-center greedy on a random projection. Official default: an approximate
      greedy coreset.
    - Scoring: max patch distance. The paper's image-score reweighting is omitted.
Expect results in the same range as published PatchCore numbers, not identical to them.

Pipeline
    1. Frozen ImageNet WideResNet50-2, features from layer2 (512ch, 28x28) and layer3 (1024ch, 14x14).
    2. Local aggregation: 3x3 average pooling so each location describes its neighbourhood.
    3. Upsample layer3 to 28x28 and concatenate -> 1536-dim feature per patch, 784 patches per image.
    4. Memory bank of all training patches, reduced with greedy (k-center) coreset selection.
    5. Patch anomaly score = distance to the nearest memory-bank feature.
       Image score = max patch score. Anomaly map = upsampled + Gaussian-blurred patch scores.

Every memory-bank entry remembers (train_image_idx, row, col), which powers the
nearest-normal-patch explanation in explain.py.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.ndimage import gaussian_filter

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """WideResNet50-2 truncated after layer3; returns locally aggregated layer2+layer3 features."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2, f3 = self.pool(f2), self.pool(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)          # (B, 1536, 28, 28) for 224x224 input


def to_tensor(images):
    """uint8 (B,H,W,3) -> normalized float tensor (B,3,H,W)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def extract(extractor, images, device, batch_size=16):
    """Patch features for a stack of images -> float32 tensor (B, 28, 28, 1536) on CPU."""
    extractor.to(device)
    out = []
    for i in range(0, len(images), batch_size):
        feats = extractor(to_tensor(images[i:i + batch_size]).to(device))
        out.append(feats.permute(0, 2, 3, 1).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def greedy_coreset(features, ratio, projection_dim=128, device="cpu", seed=0):
    """k-center greedy selection (as in PatchCore). Distances use a random projection for speed.

    Each step picks the point farthest from everything already selected, so the kept subset
    covers the feature space instead of over-representing common patches.
    """
    n = features.shape[0]
    n_select = max(1, int(n * ratio))
    if n_select >= n:
        return torch.arange(n)
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(features.shape[1], projection_dim, generator=gen) / projection_dim ** 0.5
    z = torch.cat([(chunk.to(device).float() @ proj.to(device))
                   for chunk in features.split(65536)])
    selected = torch.empty(n_select, dtype=torch.long)
    idx = int(torch.randint(n, (1,), generator=gen))
    min_dist = torch.full((n,), float("inf"), device=device)
    for k in range(n_select):
        selected[k] = idx
        d = torch.linalg.norm(z - z[idx], dim=1)
        min_dist = torch.minimum(min_dist, d)
        idx = int(torch.argmax(min_dist))
    return selected


@torch.inference_mode()
def nearest_neighbour(queries, bank, device, chunk=4096):
    """For each query row, distance to and index of the nearest bank row."""
    bank = bank.to(device).float()
    dists, idxs = [], []
    for q in queries.split(chunk):
        d = torch.cdist(q.to(device).float(), bank)
        dmin, imin = d.min(dim=1)
        dists.append(dmin.cpu())
        idxs.append(imin.cpu())
    return torch.cat(dists), torch.cat(idxs)


class PatchCore:
    def __init__(self, pretrained=True, device=None, image_size=224, blur_sigma=4, extractor=None):
        self.device = torch.device(device) if device is not None else pick_device()
        # an extractor can be shared between models (the demo serves 6 categories with one)
        self.extractor = (extractor or FeatureExtractor(pretrained=pretrained)).to(self.device)
        self.image_size = image_size
        self.blur_sigma = blur_sigma
        self.bank = None          # (M, 1536) float16
        self.bank_meta = None     # (M, 3) int32: train_image_idx, row, col
        self.grid = None          # feature grid size, 28 for 224 input
        self.train_paths = []     # source images of bank entries (for explanations)
        self.bank_crops = None    # optional (M, c, c, 3) uint8 crops, used by the offline demo

    # -- training -------------------------------------------------------------------------------
    def fit(self, images, coreset_ratio=0.1, projection_dim=128, batch_size=16, seed=0,
            train_paths=None):
        feats = extract(self.extractor, images, self.device, batch_size)   # (N, g, g, C)
        return self.fit_from_features(feats, coreset_ratio, projection_dim, seed, train_paths)

    def extract_cached(self, images, cache_path=None, batch_size=16):
        """Extract features, or load them from cache_path if it exists (float16 on disk).

        Lets a run that disconnects after extraction resume at coreset selection, and lets the
        coreset ablation reuse one extraction for every ratio.
        """
        if cache_path is not None and Path(cache_path).exists():
            feats = torch.load(cache_path, map_location="cpu")
            if len(feats) == len(images):
                return feats.float()
        feats = extract(self.extractor, images, self.device, batch_size)
        if cache_path is not None:
            tmp = Path(str(cache_path) + ".tmp")
            torch.save(feats.half(), tmp)
            tmp.replace(cache_path)          # atomic: never leaves a half-written cache file
        return feats

    @property
    def device_name(self):
        if self.device.type == "cuda":
            return f"cuda ({torch.cuda.get_device_name(self.device)})"
        return "cpu"

    def fit_from_features(self, feats, coreset_ratio, projection_dim=128, seed=0, train_paths=None):
        """Same as fit() but reuses already-extracted features (used by the coreset ablation)."""
        n, g, _, c = feats.shape
        self.grid = g
        flat = feats.reshape(-1, c)
        img_idx, rows, cols = np.meshgrid(np.arange(n), np.arange(g), np.arange(g), indexing="ij")
        meta = np.stack([img_idx.ravel(), rows.ravel(), cols.ravel()], axis=1).astype(np.int32)
        keep = greedy_coreset(flat, coreset_ratio, projection_dim, self.device, seed).numpy()
        self.bank = flat[keep].half()
        self.bank_meta = meta[keep]
        self.train_paths = [str(p) for p in (train_paths or [])]
        return self

    # -- inference ------------------------------------------------------------------------------
    def score_features(self, feats):
        n, g, _, c = feats.shape
        dist, nn_idx = nearest_neighbour(feats.reshape(-1, c), self.bank, self.device)
        patch_scores = dist.reshape(n, g, g).numpy()
        nn_idx = nn_idx.reshape(n, g, g).numpy()
        maps = self.patch_to_map(patch_scores)
        image_scores = patch_scores.reshape(n, -1).max(axis=1)
        return {"image_scores": image_scores, "maps": maps,
                "patch_scores": patch_scores, "nn_idx": nn_idx}

    def predict(self, images, batch_size=16):
        """Score images. Returns dict with image_scores, maps (N,H,W), patch_scores, nn_idx,
        and seconds_per_image (feature extraction + kNN, excluding image loading)."""
        t0 = time.perf_counter()
        feats = extract(self.extractor, images, self.device, batch_size)
        out = self.score_features(feats)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        out["seconds_per_image"] = (time.perf_counter() - t0) / max(1, len(images))
        return out

    def patch_to_map(self, patch_scores):
        t = torch.from_numpy(patch_scores).float().unsqueeze(1)
        up = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear",
                           align_corners=False).squeeze(1).numpy()
        return np.stack([gaussian_filter(m, sigma=self.blur_sigma) for m in up]).astype(np.float32)

    # -- persistence ----------------------------------------------------------------------------
    def save(self, path):
        torch.save({"bank": self.bank, "bank_meta": self.bank_meta, "grid": self.grid,
                    "train_paths": self.train_paths, "bank_crops": self.bank_crops,
                    "image_size": self.image_size, "blur_sigma": self.blur_sigma}, path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.bank = state["bank"]
        self.bank_meta = state["bank_meta"]
        self.grid = state["grid"]
        self.train_paths = state["train_paths"]
        self.bank_crops = state.get("bank_crops")
        self.image_size = state["image_size"]
        self.blur_sigma = state["blur_sigma"]
        return self

    @property
    def bank_megabytes(self):
        return self.bank.numel() * self.bank.element_size() / 2 ** 20
