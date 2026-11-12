"""Overlays and figures (used by the evaluation scripts and the Streamlit app)."""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def heatmap_overlay(image, anomaly_map, vmin, vmax, alpha=0.5):
    norm = np.clip((anomaly_map - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * image).astype(np.uint8)


def draw_defects(image, mask, defects, gt_mask=None):
    out = image.copy()
    if gt_mask is not None and gt_mask.any():
        contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 1)          # ground truth: green
    tint = out.copy()
    tint[mask.astype(bool)] = (255, 0, 0)
    out = (0.65 * out + 0.35 * tint).astype(np.uint8)
    for i, d in enumerate(defects, 1):
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 230, 0), 2)
        cv2.putText(out, f"#{i}", (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 230, 0), 1, cv2.LINE_AA)
    return out


def qualitative_grid(rows, path, title=None):
    """rows: list of dicts with keys 'image', 'gt', 'pc_overlay', 'cl_overlay', 'label'."""
    cols = ["image", "gt", "pc_overlay", "cl_overlay"]
    names = ["Input", "Ground truth", "PatchCore", "Classical"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.6 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            img = row[key]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=10)
        axes[r, 0].set_ylabel(row.get("label", ""), fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def metric_bars(df, metric, path):
    """Grouped bar chart: categories on x, one bar per method."""
    pivot = df.pivot(index="category", columns="method", values=metric)
    ax = pivot.plot(kind="bar", figsize=(8, 4), rot=0, width=0.75)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.legend(title=None, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{metric} by category")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def line_plot(df, x, y, hue, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, sub in df.groupby(hue):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], marker="o", label=str(name))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
