# Pneumonia Diagnosis Detection and Localization

This project classifies chest X-rays as pneumonia / no pneumonia and produces a weakly supervised Grad-CAM localization. RSNA bounding boxes are retained for evaluation only; they are not classifier training targets.

## Dataset

The RSNA Pneumonia Detection Challenge files are supplied separately and must remain local. Expected files are `stage_2_train_images/`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` under `data/raw/` or one dataset folder beneath it. The project never downloads, alters, or commits the original dataset. `.gitignore` excludes `data/`, DICOMs, archives, checkpoints, results, and Python environments.

The labels contain 30,227 annotation rows and 26,684 unique images: 6,012 positive and 20,672 negative image IDs after grouping box rows. There are 3,398 images with multiple annotation rows. The local `stage_2_train_images/` directory contains 26,684 DICOM files, matching the label IDs. The separate 3,000-image `stage_2_test_images/` folder is unlabeled and is not used for this experiment.

## Methodology and architecture

- Group every label row and bounding box by image ID before sampling.
- Select a stratified image-level subset of at most 26,684 IDs (the available labeled training set), then split reproducibly 80/10/10 using seed 42.
- DenseNet121 produces a single pneumonia logit. Optional ImageNet initialization is requested with `--pretrained`.
- Train using image-level pneumonia labels and weighted binary cross-entropy. Bounding boxes are not training targets.
- Grad-CAM from the final DenseNet feature normalization layer produces a heatmap and candidate boxes using adaptive thresholding, morphological cleanup, and connected components.
- Classification is evaluated on all held-out test images. Localization is evaluated for pneumonia-positive test images with boxes; the primary score assigns IoU 0 to classifier false negatives. A classifier-independent localization score is also reported.

The reference paper is summarized in [`docs/reference_paper_analysis.md`](docs/reference_paper_analysis.md). Its reported metrics are not this project's results.

## Setup

Python 3.10+ is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied dataset should be located at `data/raw/<dataset-folder>/stage_2_train_images/` and contain matching `.dcm` files for label IDs. Confirm locally with:

```bash
find data/raw -name stage_2_train_labels.csv -o -name '*.dcm' | head
```

## Training command

The preserved baseline uses all 26,684 labeled images, seed 42, an 80/10/10 train/validation/test split, two epochs, 128×128 inputs, batch size 32, and ImageNet-pretrained DenseNet121. `--pretrained` requests ImageNet weights and may need network access or a cached torchvision weight file; omit it for random initialization if weights cannot be obtained.

```bash
python -m src.train --data-root data/raw --max-images 26684 --seed 42 --epochs 2 --batch-size 32 --image-size 128 --pretrained
```

The selected IDs/split are recorded at `results/selection_manifest.csv`; the baseline checkpoint is `checkpoints/best_model.pt`. Supported subset sizes up to 26,684 include 1000, 5000, 10000, 20000, and 26684. A request above the cap fails clearly.

The planned next experiment uses the same DenseNet121/AdamW setup with 224×224 inputs, up to 10 epochs, and early stopping after three epochs without validation-loss improvement:

```bash
python -m src.train --data-root data/raw --max-images 26684 --seed 42 --epochs 10 --patience 3 --batch-size 8 --image-size 224 --pretrained --checkpoint checkpoints/densenet121_224_ep10.pt
```

## Evaluation command

```bash
python -m src.evaluate --data-root data/raw --max-images 26684 --seed 42 --checkpoint checkpoints/best_model.pt --results results/baseline_tuned --batch-size 8
```

Evaluation compares validation thresholds 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, and 0.85, selects the best F1 (ties by sensitivity, then specificity), and applies it unchanged to test. It reports classification metrics, threshold comparison, and localization mean/median IoU and rates at IoU ≥ 0.3/0.5, with classifier-conditioned and independent results, under the specified results directory.

## Streamlit demo

After installing requirements and training the checkpoint:

```bash
streamlit run app.py
```

Upload PNG, JPG, or DICOM to display pneumonia probability, binary prediction, Grad-CAM, and the predicted heatmap box. A checkpoint is required.

## Device selection

The code selects CUDA if available, then Apple MPS using `torch.backends.mps.is_available()`, then CPU.

## Actual final results and current limitations

Run date: 2026-10-08. The baseline uses all 26,684 labeled images (6,012 positive, 20,672 negative), seed 42; split counts were train 21,347 (4,783 positive / 16,564 negative), validation 2,668 (626 / 2,042), and test 2,669 (603 / 2,066). It used two epochs, 128×128 inputs, batch size 32, ImageNet-pretrained DenseNet121, and AdamW at 1e-4. The best checkpoint was epoch 2 (validation loss 0.6925). The prior accuracy-selected threshold was 0.8464. New threshold selection uses validation F1.

Validation threshold comparison (threshold, recall, precision, F1, specificity, accuracy):

| Threshold | Recall | Precision | F1 | Specificity | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 0.9473 | 0.3943 | 0.5568 | 0.5539 | 0.6462 |
| 0.40 | 0.9026 | 0.4261 | 0.5789 | 0.6273 | 0.6919 |
| 0.50 | 0.8658 | 0.4734 | 0.6121 | 0.7047 | 0.7425 |
| 0.60 | 0.8051 | 0.5333 | 0.6416 | 0.7840 | 0.7890 |
| **0.70** | **0.7188** | **0.5960** | **0.6517** | **0.8506** | **0.8197** |
| 0.80 | 0.5815 | 0.6842 | 0.6287 | 0.9177 | 0.8388 |
| 0.85 | 0.4856 | 0.7451 | 0.5880 | 0.9491 | 0.8403 |

Held-out classification results at the validation-selected probability threshold 0.70:

| Metric | Result |
|---|---:|
| Accuracy | 0.8202 |
| Precision | 0.5834 |
| Recall / sensitivity | 0.7131 |
| Specificity | 0.8514 |
| F1 | 0.6418 |
| AUROC | 0.8697 |
| Confusion matrix (actual rows [negative, positive], predicted columns [negative, positive]) | `[[1759, 307], [173, 430]]` |

Adaptive, cleaned multi-component Grad-CAM boxes were evaluated on 603 pneumonia-positive test images with boxes. Primary classifier-conditioned mean IoU was **0.1038** (median 0.0645), with **9.95%** at IoU ≥ 0.3 and **0.50%** at IoU ≥ 0.5. Classifier-independent mean IoU was **0.1187** (median 0.0775), with **10.28%** at IoU ≥ 0.3 and **0.50%** at IoU ≥ 0.5. This extraction rule did not improve mean IoU over the prior single-box baseline; localization remains weak.

Generated artifacts: `checkpoints/best_model.pt`; `results/metrics.json`; `results/localization_metrics.json`; `results/confusion_matrix.svg`; `results/roc_curve.svg`; `results/selection_manifest.csv`; and high-resolution PNGs under `results/examples/`. The results folder is ignored by Git to keep generated artifacts and dataset-related outputs out of the public repository.

Limitations: test accuracy at the F1-oriented operating point is 82.0%; sensitivity rose from 48.3% to 71.3%, while specificity fell to 85.1%. Adaptive component localization remains poor. The planned 224×224, 10-epoch training was started but stopped before completing an epoch because this runtime only exposed CPU and a full-dataset run was not practical. The images came from the RSNA training partition, so this is not external validation; Grad-CAM boxes are coarse explanations rather than validated lesion segmentations; and the reference paper leaves its CAM IoU interpretation ambiguous.

## PPT and report

[`docs/PPT_OUTLINE.md`](docs/PPT_OUTLINE.md) contains the requested 16-slide content and actual results. [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) contains the two-page report with the same run results and limitations.
