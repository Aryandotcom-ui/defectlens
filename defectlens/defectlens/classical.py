"""Classical baseline: handcrafted Gabor + LBP descriptors with per-location Gaussians.

PaDiM-inspired (Defard et al., 2021): PaDiM fits a Gaussian per spatial location to CNN
embeddings and scores with Mahalanobis distance. Here the CNN embeddings are replaced by
handcrafted texture descriptors, so this is "PaDiM-inspired", not PaDiM.

Descriptor at each location of a 28x28 grid (16x16 window, stride 8, on a 224x224 image):
    - Gabor bank, 4 orientations x 3 wavelengths = 12 filters (quadrature magnitude):
      mean and std of each response in the window  -> 24 dims
    - Uniform LBP (P=8, R=1), 10-bin histogram     -> 10 dims
    Total 34 dims.

Covariances use Ledoit-Wolf shrinkage so they stay well-conditioned (LBP histograms sum to
1, which makes the plain sample covariance singular).
"""

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern
from sklearn.covariance import LedoitWolf

GRID_STRIDE = 8   # window 16 = 2x2 blocks of 8x8, giving a 28x28 grid like PatchCore


def gabor_bank(wavelengths=(4, 8, 16), n_orientations=4):
    """Quadrature pairs (even, odd) for each (wavelength, orientation)."""
    bank = []
    for lam in wavelengths:
        sigma = 0.56 * lam                       # ~1 octave bandwidth
        ksize = int(6 * sigma) | 1
        for k in range(n_orientations):
            theta = np.pi * k / n_orientations
            even = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            odd = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, np.pi / 2,
                                     ktype=cv2.CV_32F)
            even -= even.mean()                  # zero-DC so flat regions respond with 0
            bank.append((even, odd))
    return bank


def _window_mean(channels):
    """Mean over 16x16 windows with stride 8 for an image padded to (g+1)*8.

    channels: (H, W, C) with H = W = (g+1)*8. Uses 8x8 block sums, then sums each 2x2 block
    neighbourhood, which is exactly a 16x16 window at stride 8 -> (g, g, C).
    """
    h, w, c = channels.shape
    b = GRID_STRIDE
    blocks = channels.reshape(h // b, b, w // b, b, c).sum(axis=(1, 3))
    win = blocks[:-1, :-1] + blocks[1:, :-1] + blocks[:-1, 1:] + blocks[1:, 1:]
    return win / (4 * b * b)


class ClassicalDetector:
    def __init__(self, image_size=224, wavelengths=(4, 8, 16), n_orientations=4,
                 lbp_points=8, lbp_radius=1, blur_sigma=4):
        self.image_size = image_size
        self.wavelengths = tuple(wavelengths)
        self.n_orientations = n_orientations
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.blur_sigma = blur_sigma
        self.bank = gabor_bank(self.wavelengths, n_orientations)
        self.n_lbp_bins = lbp_points + 2
        self.feat_mean = self.feat_std = None     # global standardization
        self.loc_mean = self.loc_prec = None      # (L, D), (L, D, D)
        self.grid = None

    # -- descriptors ----------------------------------------------------------------------------
    def describe(self, image):
        """RGB uint8 (S,S,3) -> descriptor grid (g, g, 34)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        pad = GRID_STRIDE // 2
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

        mags = []
        for even, odd in self.bank:
            re = cv2.filter2D(gray, cv2.CV_32F, even, borderType=cv2.BORDER_REFLECT_101)
            im = cv2.filter2D(gray, cv2.CV_32F, odd, borderType=cv2.BORDER_REFLECT_101)
            mags.append(np.sqrt(re * re + im * im))
        mags = np.stack(mags, axis=-1)                                  # (H, W, 12)
        mean = _window_mean(mags)
        sq = _window_mean(mags * mags)
        std = np.sqrt(np.clip(sq - mean * mean, 0, None))

        lbp = local_binary_pattern((gray * 255).astype(np.uint8), self.lbp_points,
                                   self.lbp_radius, method="uniform").astype(np.int64)
        onehot = np.eye(self.n_lbp_bins, dtype=np.float32)[lbp]        # (H, W, 10)
        hist = _window_mean(onehot)                                     # normalized histogram
        return np.concatenate([mean, std, hist], axis=-1).astype(np.float32)

    def describe_many(self, images):
        return np.stack([self.describe(im) for im in images])

    @property
    def feature_groups(self):
        """Index ranges of descriptor groups, used for per-group explanations."""
        n = len(self.bank)
        groups = {}
        for i, lam in enumerate(self.wavelengths):
            for k in range(self.n_orientations):
                j = i * self.n_orientations + k
                groups.setdefault(f"Gabor {int(180 * k / self.n_orientations)}°", []).extend(
                    [j, n + j])
        groups["LBP texture"] = list(range(2 * n, 2 * n + self.n_lbp_bins))
        return groups

    # -- fitting --------------------------------------------------------------------------------
    def fit(self, images):
        feats = self.describe_many(images)                              # (N, g, g, D)
        n, g, _, d = feats.shape
        self.grid = g
        flat = feats.reshape(n, g * g, d)
        self.feat_mean = flat.reshape(-1, d).mean(axis=0)
        self.feat_std = flat.reshape(-1, d).std(axis=0) + 1e-6
        flat = (flat - self.feat_mean) / self.feat_std
        self.loc_mean = flat.mean(axis=0)
        self.loc_prec = np.empty((g * g, d, d), np.float32)
        for loc in range(g * g):
            lw = LedoitWolf().fit(flat[:, loc, :])
            self.loc_prec[loc] = lw.precision_
        return self

    # -- inference ------------------------------------------------------------------------------
    def score_features(self, feats):
        n, g, _, d = feats.shape
        x = (feats.reshape(n, g * g, d) - self.feat_mean) / self.feat_std
        diff = x - self.loc_mean[None]
        m2 = np.einsum("nld,lde,nle->nl", diff, self.loc_prec, diff)
        patch_scores = np.sqrt(np.clip(m2, 0, None)).reshape(n, g, g).astype(np.float32)
        maps = self.patch_to_map(patch_scores)
        return {"image_scores": patch_scores.reshape(n, -1).max(axis=1), "maps": maps,
                "patch_scores": patch_scores}

    def predict(self, images):
        t0 = time.perf_counter()
        out = self.score_features(self.describe_many(images))
        out["seconds_per_image"] = (time.perf_counter() - t0) / max(1, len(images))
        return out

    def patch_to_map(self, patch_scores):
        t = torch.from_numpy(patch_scores).float().unsqueeze(1)
        up = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear",
                           align_corners=False).squeeze(1).numpy()
        return np.stack([gaussian_filter(m, sigma=self.blur_sigma) for m in up]).astype(np.float32)

    def group_deviation(self, image, row, col):
        """Mean |z-score| per descriptor group at one grid cell (a simple explanation)."""
        f = (self.describe(image)[row, col] - self.feat_mean) / self.feat_std
        loc = row * self.grid + col
        z = np.abs(f - self.loc_mean[loc])
        return {name: float(z[idx].mean()) for name, idx in self.feature_groups.items()}

    # -- persistence ----------------------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(
            path, feat_mean=self.feat_mean, feat_std=self.feat_std, loc_mean=self.loc_mean,
            loc_prec=self.loc_prec, grid=self.grid, image_size=self.image_size,
            wavelengths=np.array(self.wavelengths), n_orientations=self.n_orientations,
            lbp_points=self.lbp_points, lbp_radius=self.lbp_radius, blur_sigma=self.blur_sigma)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        det = cls(int(z["image_size"]), tuple(int(w) for w in z["wavelengths"]),
                  int(z["n_orientations"]), int(z["lbp_points"]), int(z["lbp_radius"]),
                  float(z["blur_sigma"]))
        det.feat_mean, det.feat_std = z["feat_mean"], z["feat_std"]
        det.loc_mean, det.loc_prec = z["loc_mean"], z["loc_prec"]
        det.grid = int(z["grid"])
        return det
