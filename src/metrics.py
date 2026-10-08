"""Dependency-light binary and localization metrics plus PPT-ready plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def binary_metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: float(a / b) if b else 0.0
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall)
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos and n_neg:
        pos_scores, neg_scores = p[positives], p[~positives]
        wins = (pos_scores[:, None] > neg_scores[None, :]).sum()
        ties = (pos_scores[:, None] == neg_scores[None, :]).sum()
        auroc = float((wins + 0.5 * ties) / (n_pos * n_neg))
    else:
        auroc = None
    return {
        "count": len(y), "threshold": threshold,
        "accuracy": div(tp + tn, len(y)), "precision": precision,
        "recall_sensitivity": recall, "specificity": specificity, "f1": f1,
        "auroc": auroc, "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def best_accuracy_threshold(labels, probabilities):
    """Choose the accuracy-maximizing threshold using validation data only."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(y):
        return 0.5, 0.0
    thresholds = np.unique(p)[::-1]
    accuracies = np.asarray([np.mean((p >= threshold).astype(np.int64) == y) for threshold in thresholds])
    best = int(accuracies.argmax())
    return float(thresholds[best]), float(accuracies[best])


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def heatmap_box(cam, original_size, threshold=0.65):
    width, height = original_size
    cam = cam.detach().cpu().numpy()
    binary = cam >= max(threshold, float(np.quantile(cam, 0.85)))
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return [0, 0, width, height]
    x0, x1 = xs.min() * width / cam.shape[1], (xs.max() + 1) * width / cam.shape[1]
    y0, y1 = ys.min() * height / cam.shape[0], (ys.max() + 1) * height / cam.shape[0]
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def save_visual(original, cam, pred_box, gt_boxes, path, title=""):
    original = original.convert("RGB")
    overlay = original.copy()
    heat = Image.fromarray(np.uint8(np.clip(cam.detach().cpu().numpy(), 0, 1) * 255)).resize(original.size)
    heat_rgb = Image.new("RGB", original.size, (255, 0, 0))
    overlay = Image.blend(overlay, Image.composite(heat_rgb, Image.new("RGB", original.size), heat), 0.40)
    panels = []
    for image, draw_boxes in ((original.copy(), True), (overlay, True)):
        draw = ImageDraw.Draw(image)
        if draw_boxes:
            x, y, w, h = pred_box
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=3)
            for box in gt_boxes:
                x, y, w, h = box
                draw.rectangle((x, y, x + w, y + h), outline=(255, 40, 40), width=3)
        panels.append(image)
    canvas = Image.new("RGB", (original.width * 2, original.height + 32), "white")
    canvas.paste(panels[0], (0, 32)); canvas.paste(panels[1], (original.width, 32))
    ImageDraw.Draw(canvas).text((8, 8), title or "Original (green prediction, red ground truth) | Grad-CAM overlay", fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_confusion_svg(matrix, path):
    tn, fp = matrix[0]; fn, tp = matrix[1]
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="500" height="390"><rect width="100%" height="100%" fill="white"/><text x="250" y="32" text-anchor="middle" font-size="20">Confusion matrix (actual rows, predicted columns)</text><text x="180" y="80">Predicted no</text><text x="320" y="80">Predicted yes</text><text x="30" y="165">Actual no</text><text x="30" y="265">Actual yes</text><rect x="150" y="100" width="120" height="90" fill="#b7e4c7"/><rect x="270" y="100" width="120" height="90" fill="#ffd6a5"/><rect x="150" y="190" width="120" height="90" fill="#ffd6a5"/><rect x="270" y="190" width="120" height="90" fill="#b7e4c7"/><g font-size="24" text-anchor="middle"><text x="210" y="155">{tn}</text><text x="330" y="155">{fp}</text><text x="210" y="245">{fn}</text><text x="330" y="245">{tp}</text></g></svg>'''
    Path(path).write_text(text, encoding="utf-8")


def save_roc_svg(labels, probabilities, path):
    y = np.asarray(labels); p = np.asarray(probabilities)
    thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]
    points=[]
    positives=max(1,int((y==1).sum())); negatives=max(1,int((y==0).sum()))
    for t in thresholds:
        pred=p>=t
        points.append((float((pred & (y==0)).sum())/negatives, float((pred & (y==1)).sum())/positives))
    coords=" ".join(f"{60+x*360:.1f},{320-yv*260:.1f}" for x,yv in points)
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="390"><rect width="100%" height="100%" fill="white"/><text x="250" y="28" text-anchor="middle" font-size="20">ROC curve</text><line x1="60" y1="320" x2="420" y2="60" stroke="#aaa" stroke-dasharray="5,5"/><polyline points="{coords}" fill="none" stroke="#1769aa" stroke-width="4"/><text x="240" y="370" text-anchor="middle">False positive rate</text><text x="18" y="190" transform="rotate(-90 18,190)">True positive rate</text></svg>',encoding='utf-8')


def write_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
