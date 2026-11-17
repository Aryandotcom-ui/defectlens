# DefectLens — Explainable Industrial Defect Inspection

DefectLens learns what *good* products look like from defect-free images only, then inspects new
images: it flags anomalies, localizes them with pixel-level heatmaps, turns them into bounding
boxes and a PASS/FAIL inspection report, and shows **why** a region was flagged by retrieving the
closest normal patch it was compared against.

Two detectors are compared on MVTec AD (carpet, leather, tile, bottle, screw, metal_nut):

| | Method A: PatchCore | Method B: Classical baseline |
|---|---|---|
| Features | ImageNet WideResNet50-2, layer2 + layer3, 3×3 local aggregation | Gabor bank (4 orientations × 3 scales) + uniform LBP |
| Normal model | Greedy-coreset memory bank of normal patches | Gaussian per grid location, Ledoit-Wolf shrinkage |
| Score | Nearest-neighbour distance | Mahalanobis distance |
| Reference | Custom PatchCore-style detector, **implemented from scratch**, after Roth et al., CVPR 2022 | PaDiM-inspired (Defard et al., 2021), handcrafted features |

## Results

MVTec AD, 6 categories, 224×224 input. Thresholds calibrated on held-out normal images only.

**Mean over the 6 categories**

| Method | Image AUROC | Pixel AUROC | AU-PRO (30% FPR) | Image F1 | Speed, same CPU |
|---|---|---|---|---|---|
| PatchCore-style (ours) | 0.985 | 0.984 | 0.925 | 0.966 | 903 ms/img (43 ms on a T4 GPU) |
| Classical (Gabor+LBP) | 0.887 | 0.931 | 0.794 | 0.753 | 77 ms/img |

**Per category: Image AUROC / AU-PRO**

| Category | PatchCore-style | Classical |
|---|---|---|
| carpet | 0.998 / 0.937 | 0.748 / 0.698 |
| leather | 1.000 / 0.965 | 0.999 / 0.954 |
| tile | 1.000 / 0.847 | 0.956 / 0.679 |
| bottle | 1.000 / 0.933 | 0.970 / 0.888 |
| screw | 0.917 / 0.934 | 0.804 / 0.851 |
| metal_nut | 0.997 / 0.932 | 0.847 / 0.695 |

**Findings**
- Deep patch features beat handcrafted texture features overall, by the widest margin on carpet and on the rotated parts (screw, metal_nut).
- The classical baseline is near-perfect on leather, so handcrafted features suffice for some textures.
- Per-location statistics assume alignment, and break on parts that appear at arbitrary rotations (classical recall on screw: 0.18).
- Speed–accuracy trade-off: on the same CPU the classical baseline is ~12× faster.
- AU-PRO matches MVTec's reference implementation (max |diff| 3.9e-4 across all 12 maps).

**Live demo:** `**Live demo:** https://https://defectlens.streamlit.app`

---

## How the pieces fit

```
 GitHub (code) ──git clone──► Google Colab (GPU/CPU compute) ◄──mount──► Google Drive (data, caches, results)
                                       │
                                       └──upload──► Hugging Face Space (public Streamlit demo)
```

Nothing heavy runs on your laptop. Its only job is a browser.

| Service | What it does here | Cost |
|---|---|---|
| **GitHub** | Holds the code. Every notebook `git clone`s it, so all runs use the same code. | Free |
| **Google Colab** | Runs the code on Google's machines, with a free T4 GPU for PatchCore. Sessions are temporary: anything not saved to Drive is lost when the session ends. | Free (GPU time is quota-limited) |
| **Google Drive** | Permanent storage mounted into Colab: the dataset, model files, scores and results. Every script skips finished work, so after a disconnect you simply re-run the notebook. | Free 15 GB (we use ~5–6 GB incl. feature cache) |
| **Kaggle Notebooks** | Backup GPU if Colab's quota runs out. Same scripts, different paths (see below). | Free weekly GPU hours |
| **Hugging Face Spaces** | Hosts the Streamlit demo on a free CPU machine and gives you a public URL for your resume. Built from a Docker image (`deploy/Dockerfile`). | Free CPU tier |

## Quickstart (all in the browser)

1. **Push this repo to GitHub** (public or private; for private, use a token in the clone URL).
2. **Open `notebooks/00_setup_and_data.ipynb` in Colab** (Colab → File → Open notebook → GitHub tab).
   Set `REPO_URL`, accept the MVTec AD license on the MVTec website, paste the download link,
   run all cells. The 6 categories land in `MyDrive/defectlens/data/mvtec_ad/`.
3. **Open `notebooks/run_all.ipynb`** on a **T4 GPU** runtime and run its steps in order:
   1. PatchCore → memory banks + scores
   2. Classical baseline → models + scores (runs on CPU even on the GPU runtime)
   3. Evaluate → thresholds, metrics table, figures in `MyDrive/defectlens/work/results/`
   4. AU-PRO parity check + same-device speed benchmark
   5. Show results
   6. *(Stretch)* robustness study + coreset ablation
   7. Deploy the demo to your Hugging Face Space

Both notebooks start with the same setup cell (mount Drive, clone/pull repo, install requirements).
If Colab disconnects, re-run the setup cell and continue from the step you were on; finished work is skipped.

### Running on Kaggle instead
Create a Kaggle notebook with GPU enabled, attach an MVTec AD dataset, then:
```bash
!git clone https://github.com/<you>/defectlens.git && cd defectlens && pip install -q -r requirements.txt
!python defectlens/scripts/run_patchcore.py --data_root /kaggle/input/<dataset-folder> --work_root /kaggle/working/work
```
Download `/kaggle/working/work` (or upload it to Drive) to continue elsewhere.

---

## Repository map

```
defectlens/
├── configs/default.yaml        all settings: paths, categories, coreset ratio, percentiles, ...
├── defectlens/                 the Python package (all the logic)
│   ├── config.py               loads the YAML, shared CLI flags (--data_root, --categories, --force)
│   ├── data.py                 MVTec loading, 85/15 train/calibration split, test images + masks
│   ├── patchcore.py            PatchCore: feature extractor, k-center coreset, kNN scoring, save/load
│   ├── classical.py            Gabor + LBP descriptors, per-location Ledoit-Wolf Gaussians, Mahalanobis
│   ├── calibration.py          thresholds from held-out NORMAL images only (99th percentile)
│   ├── postprocess.py          threshold → open/close → connected components → boxes → PASS/FAIL report
│   ├── explain.py              nearest-normal-patch retrieval (the "evidence" panel)
│   ├── metrics.py              image/pixel AUROC, AU-PRO (MVTec protocol), F1, IoU
│   ├── perturb.py              brightness / noise / blur / JPEG / rotation (masks rotate too)
│   └── viz.py                  heatmap overlays, defect drawings, figures
├── scripts/                    thin command-line entry points, resumable via Drive caches
│   ├── run_patchcore.py        fit + score PatchCore              → work/models, work/scores
│   ├── run_classical.py        fit + score classical baseline     → work/models, work/scores
│   ├── evaluate.py             thresholds, metrics, figures       → work/thresholds, work/results
│   ├── check_aupro_parity.py   AU-PRO vs MVTec's reference code   → results/aupro_parity.txt
│   ├── benchmark_speed.py      same-device speed comparison       → results/speed.csv
│   ├── ablation_coreset.py     (stretch) coreset ratio vs accuracy/memory/speed
│   ├── robustness.py           (stretch) metrics under perturbations
│   └── export_demo.py          1% banks + crops + gallery for the demo → work/demo_artifacts
├── tests/test_metrics.py       AU-PRO / IoU checks, no downloads needed
├── notebooks/                  00_setup_and_data (download data, once) · run_all (everything else)
├── app/streamlit_app.py        the demo: inspect tab, results tab, how-it-works tab
└── deploy/                     Hugging Face Space files: Dockerfile, requirements, Space README
```

Outputs on Drive (`MyDrive/defectlens/work/`):
`features/` cached training features (float16) · `models/` fitted models · `scores/` raw scores and maps · `thresholds/` calibrated τ values ·
`results/` metrics.csv, metrics.md, figures · `demo_artifacts/` what the Space serves.

## Design decisions worth defending

- **Normal-only training.** Real factories have few, varied defects; anomaly detection learns "normal".
- **Resize to 224 without center crop**, masks resized identically, so border defects aren't cut off.
- **Thresholds never touch the test set.** ~15% of `train/good` is held out; τ_image and τ_pixel are the
  99th percentiles of its scores. A starting rule, not a claimed optimum.
- **PatchCore-style, not a strict reproduction.** Same core design as the paper (layer2+3 features,
  local aggregation, greedy coreset, nearest-neighbour scoring), but direct 224 resize instead of
  256→224 crop, 1536-dim concatenated features instead of the official 1024-dim projection, exact
  k-center greedy, and no image-score reweighting. Expect numbers near, not equal to, published ones.
- **AU-PRO** follows the MVTec AD evaluator's algorithm (per-region overlap over 8-connected
  ground-truth regions, one curve point per distinct score, integrated to 30% FPR, normalized).
  Verified two ways: `tests/test_metrics.py` against a brute-force implementation of the definition,
  and `scripts/check_aupro_parity.py` against a published copy of MVTec's reference AU-PRO code
  (agreement within one FPR step, 1/N; result in `work/results/aupro_parity.txt`).
- **Timing is reported with its device.** `ms_per_img` in the metrics table comes from each method's
  own runtime (PatchCore on a T4, classical on CPU), so it is not a speed comparison;
  `scripts/benchmark_speed.py` times both on the same CPU (plus PatchCore on GPU) → `speed.csv`.
- **Explanations are retrieval-based evidence**, not causal claims, and spatially approximate: a grid
  cell is 8×8 px but its receptive field is much larger.
- **The classical baseline assumes alignment** (one Gaussian per location), so it's expected to fail on
  rotated parts (screw, metal_nut). That contrast is a finding, not a bug.
- **Demo vs. reported numbers:** the Space uses a 1% coreset bank (with its own recalibrated thresholds)
  to fit the free tier; reported results use the 10% bank.

## Credits and license

- MVTec AD: P. Bergmann et al., CVPR 2019 / IJCV 2021. **CC BY-NC-SA 4.0, non-commercial use only.**
- PatchCore: K. Roth et al., CVPR 2022. PaDiM: T. Defard et al., ICPR Workshops 2021.
- WideResNet50-2 ImageNet weights from torchvision.
