# Final experiment record

## Dataset and split

The final run used all **26,684** labeled RSNA images, grouped by image ID before splitting so all annotations/boxes for an image remain together. The 80/10/10 split uses seed **42**:

| Split | Images | Positive | Negative |
|---|---:|---:|---:|
| Train | 21,347 | 4,783 | 16,564 |
| Validation | 2,668 | 626 | 2,042 |
| Test | 2,669 | 603 | 2,066 |

Dataset totals: 30,227 annotation rows, 6,012 positive images, 20,672 negative images. The dataset remains local and is not included in GitHub.

## Training configuration and outcome

| Setting | Final run |
|---|---|
| Architecture | DenseNet121 |
| Initialization | ImageNet pretrained |
| Input resolution | 224×224 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Loss | Class-weighted binary cross-entropy with logits |
| Seed | 42 |
| Epoch cap | 10 |
| Early stopping patience | 3 validation epochs without improvement |
| Epochs completed | 6 |
| Best epoch | 3 |
| Best validation loss | 0.6725397 |
| Training device | Apple MPS |

The best model checkpoint is saved locally at `checkpoints/best_model.pt` (same weights as `checkpoints/densenet121_224_ep10.pt`). Checkpoints are Git-ignored.

## Threshold selection

The evaluation computes validation predictions and compares thresholds `0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85`. It selects the highest validation F1; ties are broken by sensitivity and then specificity. Threshold **0.70** was selected (validation F1 0.6662, sensitivity 0.6949, specificity 0.8800, accuracy 0.8366), then applied unchanged to the held-out test split. Test labels do not influence threshold selection.

## Test results

Classification covers all 2,669 held-out images:

| Metric | Result |
|---|---:|
| Accuracy | 0.8273 |
| Precision | 0.6044 |
| Recall / sensitivity | 0.6816 |
| Specificity | 0.8698 |
| F1 | 0.6407 |
| AUROC | 0.8815 |
| Confusion matrix | `[[1797, 269], [192, 411]]` |

Localization covers the 603 positive test images with available ground-truth boxes. Grad-CAM is adaptively thresholded, morphologically cleaned, and split into connected components. Per-image IoU is the maximum over predicted components and ground-truth boxes.

| Localization protocol | Mean IoU | Median IoU | IoU ≥ 0.3 | IoU ≥ 0.5 |
|---|---:|---:|---:|---:|
| Classifier-conditioned (missed positives receive 0) | 0.2051 | 0.1880 | 34.00% | 5.97% |
| Classifier-independent | 0.2307 | 0.2290 | 37.81% | 6.47% |

## Exact commands

Training command used:

```bash
python -m src.train --data-root data/raw --max-images 26684 --seed 42 --epochs 10 --patience 3 --batch-size 8 --image-size 224 --pretrained --checkpoint checkpoints/densenet121_224_ep10.pt
```

Evaluation command used:

```bash
python -m src.evaluate --data-root data/raw --max-images 26684 --seed 42 --checkpoint checkpoints/densenet121_224_ep10.pt --results results/densenet121_224 --batch-size 8
```

## Source-code map

The implementation is organized into package modules rather than standalone `src/cam.py`, `src/data.py`, or `src/utils.py` files:

- `src/train.py`: training loop, weighted loss, optimizer, best-checkpoint saving, and early stopping.
- `src/evaluate.py`: validation threshold selection, held-out classification evaluation, Grad-CAM localization evaluation, and artifact generation.
- `src/model.py`: DenseNet121 construction and Grad-CAM implementation.
- `src/metrics.py`: classification metrics, threshold comparison, connected-component CAM boxes, IoU, plots, and visual outputs.
- `src/data/rsna.py`: RSNA labels, DICOM reading, and dataset tensors.
- `src/data/selection.py`: grouped image-level sampling and reproducible split preparation.
- `src/device.py`: CUDA, MPS, or CPU selection.
- `app.py`: Streamlit upload, prediction, and Grad-CAM demo.

Generated metrics, plots, and examples are local under `results/densenet121_224/` and are excluded from Git. Summary comparison with the 128×128 baseline is in [`final_results.md`](final_results.md).
