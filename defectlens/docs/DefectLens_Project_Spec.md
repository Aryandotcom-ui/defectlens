# DefectLens — Explainable Industrial Defect Inspection

> Classical image processing + deep visual anomaly detection, with retrieval-based evidence and an inspection verdict.

**Course:** Image Processing & Camera Vision (IPCV)
**Status:** Specification locked — codebase written and smoke-tested; runs on Colab
**Timeline:** ~2 days (staged build, see §12)

---

## 1. One-line summary

DefectLens learns what *good* products look like from defect-free images only, then inspects new images: it flags anomalies, localizes them with pixel-level heatmaps, turns them into bounding boxes and a PASS/FAIL report, and shows **why** a region was flagged by retrieving the closest normal patch it was compared against. Two detectors are compared side by side: a deep PatchCore-style model and a handcrafted Gabor + LBP baseline.

## 2. Motivation

- In real factories, defective samples are rare and varied, so supervised defect classifiers are impractical. **One-class (normal-only) anomaly detection** is the standard industrial approach.
- Most student anomaly projects stop at a heatmap. DefectLens adds two things inspectors actually need:
  1. **Evidence**: "this region doesn't resemble anything seen in good samples; here is the closest normal example."
  2. **A verdict**: defect count, size, location and PASS/FAIL, produced with classical image processing.
- Comparing handcrafted texture features against pretrained CNN features is a meaningful IPCV question in its own right: *when do classical features suffice, and when do they break?*

## 3. Dataset

**MVTec AD** (Bergmann et al., CVPR 2019): industrial images with defect-free training sets, test sets containing normal and anomalous samples, and pixel-level ground-truth masks.

**Categories used (6 of 15):**

| Type | Category | Why it's included |
|---|---|---|
| Texture | carpet | Regular repeating texture |
| Texture | leather | Irregular natural texture |
| Texture | tile | Mixed texture with strong patterns |
| Object | bottle | Aligned object, relatively easy |
| Object | screw | Arbitrary rotation, small defects (hard case) |
| Object | metal_nut | Rotation variance, includes flipped parts |

The mix is deliberate. Textures should favour the classical baseline; rotated objects should break it. That contrast is a headline finding.

**Splits:**
- Training: MVTec `train/good`, minus a held-out calibration subset.
- **Calibration set:** ~15% of each category's `train/good` images, held out and never used to build the models. It's used only to set thresholds (§7). This reduces training data slightly and is stated explicitly in the report.
- Test: MVTec `test/*` with `ground_truth/*` masks.

**License:** CC BY-NC-SA 4.0 (non-commercial). Credit the dataset in the README and report.

## 4. System architecture

```
                 ┌──────────────────────────────────────────────┐
  input image ──►│ Preprocessing (resize 224×224, normalize)    │
                 └───────────────┬──────────────────────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
   ┌─────────────────────┐              ┌────────────────────────────┐
   │ Method A: PatchCore │              │ Method B: Gabor + LBP       │
   │ WRN50-2 features    │              │ per-location Gaussian       │
   │ memory bank + kNN   │              │ Mahalanobis scoring         │
   └─────────┬───────────┘              └─────────────┬──────────────┘
             │ anomaly map + image score               │ anomaly map + image score
             └──────────────────┬──────────────────────┘
                                ▼
          ┌──────────────────────────────────────────────┐
          │ Post-processing (classical IP)               │
          │ calibrated threshold → morphology →          │
          │ connected components → boxes, area, location │
          └──────────────────┬───────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │ Explanation: nearest-normal-patch retrieval (PatchCore)   │
    │ Inspection report: verdict, defects, evidence             │
    └──────────────────────────────────────────────────────────┘
                             ▼
                    Streamlit dashboard
```

## 5. Method A — PatchCore-style detector (own implementation)

A **custom PatchCore-style anomaly detector, implemented from scratch**, based on Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022. It keeps the paper's core design but is not a bit-for-bit reproduction of the official code. Deliberate differences: direct 224 resize (official: resize 256 → center crop 224); 1536-dim concatenated features (official: per-layer adaptive pooling to 1024 dims); exact k-center greedy (official default: approximate greedy); no image-score reweighting. Results should land near the published numbers, not exactly on them. The official repo / anomalib are used only to sanity-check that our numbers are in the expected range.

**Pipeline:**
1. **Preprocessing:** resize directly to **224×224** (no center crop, so border defects aren't cut off). ImageNet normalization. Ground-truth masks get the identical resize (nearest-neighbour).
2. **Backbone:** WideResNet50-2, ImageNet-pretrained, frozen. Features from **layer2** (512 ch, 28×28) and **layer3** (1024 ch, 14×14).
3. **Local aggregation:** 3×3 average pooling (stride 1, padding 1) on each feature map, so each location summarizes its neighbourhood.
4. **Fusion:** upsample layer3 to 28×28 (bilinear) and concatenate → **1536-dim patch features**, 784 patches per image.
5. **Memory bank:** all training patch features + metadata `(train_image_idx, row, col)` for retrieval.
6. **Greedy coreset subsampling** (k-center greedy):
   - Random projection to ~128 dims for the distance computations (Johnson–Lindenstrauss trick, as in the paper).
   - Default ratio **10%**. Ablation: 1% / 10% / 25% (stretch, §9).
7. **Scoring:**
   - Patch score = distance to nearest memory-bank feature.
   - Anomaly map = patch scores (28×28) → bilinear upsample to 224×224 → Gaussian blur (σ = 4).
   - Image score = max patch score. (The paper's reweighting step is optional and noted as a possible extension.)

## 6. Method B — Classical baseline (PaDiM-inspired, handcrafted features)

**Naming:** this is *PaDiM-inspired*. PaDiM models CNN embeddings with per-location Gaussians; we substitute handcrafted descriptors. Described this way everywhere.

**Pipeline:**
1. Grayscale, resize 224×224.
2. **Gabor filter bank:** 4 orientations (0°, 45°, 90°, 135°) × 3 wavelengths → 12 filters. Magnitude responses.
3. **LBP:** uniform LBP (P = 8, R = 1) → 10-bin histogram.
4. **Local descriptors on a 28×28 grid** (stride 8, window 16×16, same spatial grid as PatchCore):
   - Gabor: mean + std of each of the 12 responses → 24 dims
   - LBP histogram in the window → 10 dims
   - **Total: 34 dims per location**
5. **Per-location Gaussian:** at each grid location, fit mean and **Ledoit-Wolf shrinkage covariance** across training images (sklearn `LedoitWolf`), which keeps the covariance well-conditioned.
6. **Scoring:** Mahalanobis distance per location → upsample + blur → anomaly map. Image score = max.

**Expected weakness, reported as a finding:** per-location models assume alignment, so they should degrade badly on screw and metal_nut, where parts appear at arbitrary rotations. PatchCore's global memory bank is far less sensitive to this.

**"Fair comparison" wording:** both methods use the same categories, training split, calibration split, test images and resolution. They differ in feature representation *and* statistical model, so we frame it as handcrafted local features vs. pretrained deep representations, not a single-variable ablation.

## 7. Threshold calibration

No test-set tuning. Thresholds come only from the held-out **normal calibration set**.

| Threshold | Purpose | Initial rule |
|---|---|---|
| τ_image | PASS/FAIL decision | 99th percentile of calibration image scores |
| τ_pixel | Defect mask | 99th percentile of calibration pixel scores |

These are reasonable starting rules, not claimed to be optimal. Calibrated separately per category and per method.

## 8. Post-processing & inspection report (classical IP)

1. Binarize the anomaly map with τ_pixel.
2. **Morphological opening** (remove specks), then **closing** (fill small holes). Elliptical kernel, ~5×5.
3. **Connected components** with a minimum-area filter.
4. Per defect: bounding box, area (px and % of image), centroid, coarse location (3×3 grid: "top-left" … "bottom-right"), peak anomaly score.
5. **Verdict:** FAIL if image score > τ_image, otherwise PASS.

**Example report:**
```
Product: screw   Method: PatchCore   Verdict: FAIL (score 3.41 > τ 2.10)
Defects found: 2
  #1  area 340 px (0.68%)  location: top-left    peak 3.41
  #2  area  95 px (0.19%)  location: center      peak 2.58
Evidence: nearest normal patch distance 3.41 (calibration 99th pct: 2.10)
```

## 9. Explainability: nearest-normal-patch retrieval

For each detected defect region (PatchCore):
1. Take the highest-scoring patch inside the region.
2. Find its nearest memory-bank entry and look up `(train_image_idx, row, col)`.
3. Show **side by side:** the crop around the flagged location in the test image and the crop around the matched location in the normal training image, plus the feature distance relative to the calibration distribution.

**Honest caveats (stated in UI and report):**
- This is **retrieval-based evidence**, not a causal explanation of the defect.
- **Spatial precision is approximate.** Each 28×28 grid cell maps to an 8×8 pixel cell, but its receptive field is much larger, so crops (~32–48 px) indicate the region the feature describes, not exact pixels.

**Optional (classical method):** standardized deviation per feature group (Gabor orientations vs. LBP), giving explanations like "deviation driven mainly by 45° texture response."

## 10. Evaluation

**Primary (threshold-free):**
- **Image AUROC**: detection
- **Pixel AUROC**: localization
- **AU-PRO** (per-region overlap, integrated up to **30% FPR**): localization that doesn't let large defects dominate. Implemented following the MVTec AD evaluator's algorithm (8-connected regions, one curve point per distinct score, trapezoidal integration with interpolation at the limit). Verified against a brute-force implementation of the definition (`tests/test_metrics.py`) and against a published copy of MVTec's reference AU-PRO code (`scripts/check_aupro_parity.py`, agreement within one FPR step). MVTec's own evaluator is a download from mvtec.com rather than a package, hence the parity check instead of a direct import.

**Secondary (at calibrated thresholds):**
- Image-level precision, recall, F1
- Pixel-level F1 and IoU
- False positive rate on normal test images

**Efficiency:**
- Inference time per image, always reported with its device. The per-method runtime numbers (PatchCore on T4, classical on CPU) are not comparable, so a separate **same-device benchmark** (`scripts/benchmark_speed.py`) times both methods on the same CPU, with PatchCore on GPU as an extra row.
- Memory bank size (patches, MB)

**Ablation (stretch):** coreset ratio 1% / 10% / 25% vs. AUROC, AU-PRO and inference time.

**Robustness study (stretch):** apply perturbations to test images and re-evaluate:
- Brightness ±20–40%
- Gaussian noise (σ = 5, 10, 20)
- Gaussian blur (k = 3, 5, 7)
- JPEG compression (quality 90, 70, 50)
- Small rotations (±5°, ±10°), **with ground-truth masks transformed identically**

Output: degradation curves per method and per category.

**Hypotheses (to be confirmed, not assumed):**
- PatchCore beats the classical baseline overall, by the widest margin on objects.
- The classical baseline is competitive on textures.
- Rotation hurts the classical baseline far more than PatchCore.

## 11. Compute plan — cloud-first

**Goal:** keep load on the local laptop near zero. All heavy compute runs online; the laptop only needs a browser.

**Local hardware (reference only):** Lenovo, Ryzen 7 7000-series, integrated Radeon (no CUDA), 16 GB RAM, 512 GB SSD. Not used for extraction, training or evaluation.

| Stage | Where it runs | Why |
|---|---|---|
| Dataset download + unpack | Colab → Google Drive | Data never touches the laptop |
| PatchCore feature extraction, coreset | **Colab GPU (T4)** | Minutes instead of tens of minutes |
| Classical baseline (Gabor/LBP, Ledoit-Wolf) | Colab CPU runtime | Doesn't need a GPU; saves GPU quota |
| Evaluation, AU-PRO, figures | Colab | Reads cached scores from Drive |
| Robustness study, coreset ablation (stretch) | Colab or **Kaggle Notebooks** (backup GPU quota) | Spreads usage across free quotas |
| Streamlit demo | **Hugging Face Spaces** (free CPU tier) | Public link for the resume; nothing runs locally |
| Code | GitHub repo, cloned into each notebook | Single source of truth |

**Storage & persistence**
- Dataset, feature caches, memory banks and results live on **Google Drive** (`MyDrive/defectlens/`), mounted in every notebook, so a Colab disconnect loses nothing.
- PatchCore training features are cached per category in `work/features/train_<category>.pt` (float16, ~0.5–0.8 GB each) before coreset selection. A disconnect after extraction resumes at the coreset step, and the coreset ablation reuses the same features instead of re-extracting.
- Every stage writes outputs to Drive and **skips work that's already cached**. Re-running a notebook after a timeout resumes instead of restarting.
- Memory banks saved in float16; one category processed at a time to stay inside Colab RAM.

**Dataset access**
- MVTec AD requires accepting its license on the MVTec website. Accept it there, then download inside Colab straight to Drive and extract only the 6 categories.
- Alternative: a Kaggle mirror of MVTec AD (redistribution is allowed under CC BY-NC-SA), attached directly to a Kaggle notebook.

**Free-tier constraints & handling**
- Colab sessions can disconnect and GPU access is quota-limited → Drive caching + resumable stages; Kaggle as a second GPU source (weekly GPU-hour quota).
- Notebooks stay thin: `git clone` the repo, `pip install -r requirements.txt`, call `scripts/*.py`. Logic lives in the package, not in notebook cells, so the same code runs on Colab, Kaggle and Spaces.

**Demo deployment (Hugging Face Spaces)**
- The Space hosts the Streamlit app plus precomputed artifacts: memory banks (1% coreset version for the demo, to keep it light), classical Gaussian stats, thresholds, and a small gallery of test images.
- Inference on the free CPU tier: one image at a time, a few seconds each, which is fine for a demo.
- If the native Streamlit SDK isn't offered on Spaces, use the Docker template with Streamlit. Streamlit Community Cloud is a fallback but has tighter RAM limits.
- Large files go through Git LFS / the HF Hub.

## 12. Build plan (staged)

**Day 1 — the results must be solid** (all runs on Colab)
0. GitHub repo + Colab notebook template (mount Drive, clone, install); dataset downloaded to Drive
1. Repo skeleton, data loader (images + masks, calibration split)
2. PatchCore: feature extraction, coreset, scoring, caching
3. Classical baseline: Gabor + LBP descriptors, Ledoit-Wolf Gaussians, Mahalanobis
4. Threshold calibration
5. Evaluation on all 6 categories (AU-PRO per MVTec protocol), results tables
6. Sanity-check PatchCore numbers against published / anomalib ranges

**Day 2 — explanation, product, write-up**
1. Post-processing + inspection report
2. Nearest-normal-patch retrieval
3. Streamlit dashboard, deployed to Hugging Face Spaces
4. Figures: heatmaps, qualitative grids, per-category bars
5. README + project report

**Only if time remains:** robustness study → coreset ablation → classical feature-group explanations.

**Fallback rule:** if Day 2 runs short, cut polish, never results. Strong metrics + a working basic demo beats a pretty dashboard with weak experiments.

## 13. Streamlit dashboard

- **Sidebar:** category, method (PatchCore / Classical / Both), threshold display
- **Upload or pick a test image** (sample gallery from MVTec test set)
- **Main view:**
  - Original | heatmap overlay | binary mask + bounding boxes
  - Side-by-side comparison when "Both" is selected
  - Inspection report card (verdict, defect table)
  - **Evidence panel:** flagged crop ↔ nearest normal crop + distance
  - Ground-truth mask overlay when the image comes from the test set
- **Results tab:** metrics tables, per-category charts, (stretch) robustness curves

## 14. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep features | PyTorch, torchvision (WideResNet50-2 weights) |
| Classical IP | OpenCV (Gabor, morphology, connected components, resizing), scikit-image (LBP) |
| Statistics / ML | NumPy, SciPy, scikit-learn (LedoitWolf, ROC-AUC, metrics) |
| Evaluation | AU-PRO per the MVTec AD evaluator's algorithm (own implementation, parity-checked against MVTec's reference code) |
| Visualization | Matplotlib |
| App | Streamlit, deployed on Hugging Face Spaces |
| Compute | Google Colab (T4 GPU), Kaggle Notebooks (backup), Google Drive (storage/cache) |
| Version control | GitHub (+ Git LFS / HF Hub for large artifacts) |
| Utilities | tqdm, PyYAML (configs) |

## 15. Repository structure

```
defectlens/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # categories, image size, coreset ratio, thresholds pct
├── deploy/                       # Hugging Face Space: Dockerfile, requirements, Space README
├── defectlens/
│   ├── data.py                   # dataset, masks, calibration split
│   ├── patchcore.py              # feature extractor, coreset, memory bank, scoring
│   ├── classical.py              # Gabor + LBP descriptors, per-location Gaussians
│   ├── calibration.py            # τ_image, τ_pixel from normal calibration set
│   ├── postprocess.py            # morphology, components, report
│   ├── explain.py                # nearest-normal-patch retrieval
│   ├── metrics.py                # AUROC, AU-PRO, F1, IoU
│   ├── perturb.py                # robustness transforms (images + masks)
│   └── viz.py                    # overlays and figures
├── notebooks/
│   ├── 00_setup_and_data.ipynb   # mount Drive, clone repo, download MVTec
│   ├── 01_patchcore.ipynb        # GPU: extract, coreset, score
│   ├── 02_classical.ipynb        # CPU runtime: Gabor/LBP baseline
│   ├── 03_evaluate.ipynb         # metrics, AU-PRO, figures
│   ├── 04_stretch.ipynb          # robustness + coreset ablation
│   └── 05_deploy_space.ipynb     # export demo + push to Hugging Face
├── scripts/
│   ├── run_patchcore.py          # fit + score PatchCore
│   ├── run_classical.py          # fit + score classical baseline
│   ├── evaluate.py               # thresholds, metrics, figures
│   ├── ablation_coreset.py       # stretch
│   ├── robustness.py             # stretch
│   └── export_demo.py            # demo artifacts for the Space
├── app/
│   ├── streamlit_app.py
│   └── artifacts/                # demo memory banks, stats, thresholds, gallery (LFS)
└── docs/
    └── DefectLens_Project_Spec.md  # this file

(Data, models, scores and results live on Google Drive under MyDrive/defectlens/, not in git.)
```

## 16. Deliverables

- Working codebase (above)
- Results: per-category table (image AUROC, pixel AUROC, AU-PRO, F1, IoU, inference time) for both methods
- Figures: qualitative heatmap grids, method comparison, evidence examples, (stretch) robustness curves
- Streamlit demo
- Project report: problem, related work, methods, evaluation protocol, results, failure analysis, limitations

## 17. What we built vs. what we reused (viva honesty)

| Built ourselves | Reused |
|---|---|
| PatchCore pipeline (features, aggregation, coreset, scoring) | ImageNet-pretrained WRN50-2 weights |
| Gabor + LBP baseline with per-location Gaussians | MVTec AU-PRO algorithm (reimplemented, parity-checked) |
| Threshold calibration protocol | OpenCV / scikit-image / sklearn primitives |
| Post-processing + inspection report | MVTec AD dataset |
| Nearest-normal-patch retrieval explanation | |
| Robustness + ablation studies, dashboard | |

## 18. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Colab disconnects / GPU quota runs out | Drive caching, resumable stages; switch to Kaggle GPU |
| Colab RAM limit during coreset | Per-category processing, float16, projected distances |
| Demo too heavy for free Spaces tier | 1% coreset banks for the demo; lazy-load per category |
| Our PatchCore numbers far below published | Compare against anomalib on 1 category; check layers, pooling, blur |
| Classical baseline unstable | Ledoit-Wolf shrinkage; check descriptor scaling |
| Day 2 overruns | Stretch goals cut first; core results already done |
| Calibration thresholds too strict/loose | Report as starting rule; show the threshold sensitivity curve if time |

## 19. Likely viva questions (prep)

- Why normal-only training instead of a supervised classifier?
- Why layer2 + layer3 and not deeper layers? (deeper = more ImageNet-specific, coarser resolution)
- What does coreset subsampling approximate, and what's the trade-off?
- Why AU-PRO in addition to pixel AUROC?
- How were thresholds chosen without touching the test set?
- Why does the classical baseline fail on screw?
- Is the nearest-patch retrieval a causal explanation? (No: retrieval-based evidence, with approximate spatial precision.)
- Why Ledoit-Wolf instead of the sample covariance?

## 20. Resume line

> **DefectLens**: explainable industrial defect inspection on MVTec AD. Implemented a PatchCore-style detector (after CVPR 2022) from scratch and a PaDiM-inspired Gabor/LBP classical baseline; added retrieval-based explanations via nearest normal patches and morphological post-processing into inspection reports. Achieved X image AUROC / Y AU-PRO across 6 categories, with calibrated thresholds and robustness analysis.

## 21. References

1. K. Roth et al., "Towards Total Recall in Industrial Anomaly Detection," CVPR 2022. (PatchCore)
2. T. Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization," ICPR Workshops 2021.
3. P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
4. P. Bergmann et al., "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," IJCV 2021. (AU-PRO protocol)
5. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns," IEEE TPAMI 2002.
6. O. Ledoit, M. Wolf, "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices," Journal of Multivariate Analysis 2004.
7. S. Zagoruyko, N. Komodakis, "Wide Residual Networks," BMVC 2016.
