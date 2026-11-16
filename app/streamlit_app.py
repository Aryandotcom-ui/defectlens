"""DefectLens demo: upload or pick an image, get heatmaps, defect boxes, a PASS/FAIL report and
nearest-normal-patch evidence, for PatchCore and the classical baseline.

Run locally:  streamlit run app/streamlit_app.py
Artifacts are read from app/artifacts/ (built by scripts/export_demo.py) or $DEFECTLENS_ARTIFACTS.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from defectlens.calibration import load_thresholds  # noqa: E402
from defectlens.classical import ClassicalDetector  # noqa: E402
from defectlens.explain import explain_defects  # noqa: E402
from defectlens.patchcore import FeatureExtractor, PatchCore, pick_device  # noqa: E402
from defectlens.postprocess import find_defects, inspection_report  # noqa: E402
from defectlens.viz import draw_defects, heatmap_overlay  # noqa: E402

ART = Path(os.environ.get("DEFECTLENS_ARTIFACTS", Path(__file__).resolve().parent / "artifacts"))

st.set_page_config(page_title="DefectLens", page_icon="🔍", layout="wide")


# -- loading (cached across reruns) -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading WideResNet50-2 backbone…")
def get_extractor():
    offline = os.environ.get("DEFECTLENS_OFFLINE") == "1"   # smoke tests only
    return FeatureExtractor(pretrained=not offline).to(pick_device())


@st.cache_resource(show_spinner="Loading category models…")
def get_models(category):
    cfg = json.loads((ART / "config.json").read_text())
    d = ART / category
    pc = PatchCore(image_size=cfg["image_size"], extractor=get_extractor()).load(d / "patchcore.pt")
    cl = ClassicalDetector.load(d / "classical.npz")
    th = {"PatchCore": load_thresholds(d / "patchcore_thresholds.json"),
          "Classical": load_thresholds(d / "classical_thresholds.json")}
    return pc, cl, th


def load_config():
    path = ART / "config.json"
    if not path.exists():
        st.error(f"No demo artifacts found in `{ART}`. Run `scripts/export_demo.py` first.")
        st.stop()
    return json.loads(path.read_text())


def read_image(data, size):
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)


# -- page ---------------------------------------------------------------------------------------
cfg = load_config()
size, post = cfg["image_size"], cfg["postprocess"]
categories = [c for c in cfg["categories"] if (ART / c).exists()]

st.title("🔍 DefectLens")
st.caption("Explainable industrial defect inspection on MVTec AD — PatchCore (deep features) "
           "vs. a Gabor + LBP classical baseline, trained on defect-free images only.")

with st.sidebar:
    category = st.selectbox("Product category", categories)
    mode = st.radio("Method", ["PatchCore", "Classical", "Compare both"])
    source = st.radio("Image source", ["Gallery (MVTec test set)", "Upload"])
    st.markdown("---")
    st.caption("Dataset: MVTec AD (CC BY-NC-SA 4.0). Demo uses a 1% coreset memory bank "
               "to fit the free hosting tier; reported results use the 10% bank.")

tab_inspect, tab_results, tab_about = st.tabs(["Inspect", "Results", "How it works"])

with tab_inspect:
    image, gt_mask, caption = None, None, ""
    if source.startswith("Gallery"):
        gal = ART / category / "gallery"
        index = json.loads((gal / "index.json").read_text())
        labels = [f"{e['defect_type']} — {e['name'].split('_')[-1]}" for e in index]
        choice = st.selectbox("Test image", range(len(index)), format_func=lambda i: labels[i])
        entry = index[choice]
        image = read_image((gal / f"{entry['name']}.png").read_bytes(), size)
        gt_mask = (cv2.imread(str(gal / f"{entry['name']}_mask.png"), 0) > 127).astype(np.uint8)
        caption = f"ground truth: {entry['defect_type']}"
    else:
        up = st.file_uploader(f"Upload a '{category}' image", type=["png", "jpg", "jpeg", "bmp"])
        st.caption("Upload a photo of the same product type as the selected category; the "
                   "models only know what *normal* looks like for that category.")
        if up:
            image = read_image(up.read(), size)

    if image is not None:
        pc, cl, thresholds = get_models(category)
        methods = ["PatchCore", "Classical"] if mode == "Compare both" else [mode]
        for name in methods:
            th = thresholds[name]
            out = pc.predict(image[None]) if name == "PatchCore" else cl.predict(image[None])
            amap, score = out["maps"][0], float(out["image_scores"][0])
            mask, defects = find_defects(amap, th["tau_pixel"], post["kernel"], post["min_area"])
            if score <= th["tau_image"]:          # PASS: don't report sub-threshold blobs
                mask, defects = np.zeros_like(mask), []
            report = inspection_report(category, name, score, th["tau_image"], defects)

            st.subheader(name)
            if report["verdict"] == "FAIL":
                st.error(f"FAIL — anomaly score {score:.2f} > threshold {th['tau_image']:.2f}")
            else:
                st.success(f"PASS — anomaly score {score:.2f} ≤ threshold {th['tau_image']:.2f}")
            c1, c2, c3 = st.columns(3)
            c1.image(image, caption=f"Input ({caption})" if caption else "Input")
            c2.image(heatmap_overlay(image, amap, 0.5 * th["tau_pixel"], 2.0 * th["tau_pixel"]),
                     caption="Anomaly heatmap")
            c3.image(draw_defects(image, mask, defects, gt_mask),
                     caption="Defects (yellow boxes) · ground truth outline (green)")
            st.code(report["text"], language=None)

            if name == "PatchCore" and defects:
                st.markdown("**Evidence: nearest normal patch**")
                st.caption("Each flagged region was compared against every patch in the normal "
                           "memory bank; this is the closest one. A large distance means nothing "
                           "in the defect-free training data looks like this region. This is "
                           "retrieval-based evidence, not a causal explanation, and crop "
                           "locations are approximate (features have large receptive fields).")
                evidence = explain_defects(pc, image, out["patch_scores"][0], out["nn_idx"][0],
                                           defects, th["tau_patch"])
                for i, ev in enumerate(evidence, 1):
                    e1, e2, e3 = st.columns([1, 1, 2])
                    e1.image(cv2.resize(ev["flagged_crop"], (120, 120),
                                        interpolation=cv2.INTER_NEAREST),
                             caption=f"Defect #{i}: flagged region")
                    e2.image(cv2.resize(ev["normal_crop"], (120, 120),
                                        interpolation=cv2.INTER_NEAREST),
                             caption="Closest normal patch")
                    e3.metric("Feature distance", f"{ev['distance']:.2f}",
                              f"{ev['ratio']:.1f}× the normal 99th percentile",
                              delta_color="inverse")
            if name == "Classical" and defects:
                row, col = (int(v) for v in np.unravel_index(
                    np.argmax(out["patch_scores"][0]), out["patch_scores"][0].shape))
                dev = cl.group_deviation(image, row, col)
                st.markdown("**Which texture cues deviate most at the strongest anomaly**")
                st.bar_chart(dev, horizontal=True)
            st.markdown("---")

with tab_results:
    res = ART / "results"
    if (res / "metrics.md").exists():
        st.markdown((res / "metrics.md").read_text())
        figs = sorted((res / "figures").glob("*.png")) if (res / "figures").exists() else []
        for fig in figs:
            st.image(str(fig), caption=fig.stem)
    else:
        st.info("Results appear here after running scripts/evaluate.py before export.")

with tab_about:
    st.markdown("""
**Training data:** only defect-free images. Anything that deviates from them is an anomaly.

**PatchCore-style detector** (after Roth et al., CVPR 2022; own implementation, not a strict reproduction): ImageNet WideResNet50-2
features from layer2+layer3, 3×3 local aggregation, a greedy-coreset memory bank of normal
patches, and nearest-neighbour distance as the anomaly score.

**Classical baseline** (PaDiM-inspired): Gabor (4 orientations × 3 scales) + LBP texture
descriptors on a 28×28 grid, a Gaussian per location with Ledoit-Wolf shrinkage covariance,
Mahalanobis distance as the score.

**Thresholds** are calibrated on held-out *normal* images only (99th percentile), never on
the test set.

**Post-processing:** threshold → morphological opening/closing → connected components →
boxes, area, location → PASS/FAIL report.
""")
