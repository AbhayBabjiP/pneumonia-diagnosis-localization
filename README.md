# Pneumonia Diagnosis Detection and Localization

This project classifies chest X-rays as pneumonia / no pneumonia and produces a weakly supervised Grad-CAM localization. RSNA bounding boxes are retained for evaluation only; they are not classifier training targets.

## Dataset

The RSNA Pneumonia Detection Challenge files are supplied separately and must remain local. Expected files are `stage_2_train_images/`, `stage_2_train_labels.csv`, and `stage_2_detailed_class_info.csv` under `data/raw/` or one dataset folder beneath it. The project never downloads, alters, or commits the original dataset. `.gitignore` excludes `data/`, DICOMs, archives, checkpoints, results, and Python environments.

The labels contain 30,227 annotation rows and 26,684 unique images: 6,012 positive and 20,672 negative image IDs after grouping box rows. There are 3,398 images with multiple annotation rows. The local `stage_2_train_images/` directory contains 26,684 DICOM files, matching the label IDs. The separate 3,000-image `stage_2_test_images/` folder is unlabeled and is not used for this experiment.

## Methodology and architecture

- Group every label row and bounding box by image ID before sampling.
- Select a stratified image-level subset of at most 15,000 IDs, then split reproducibly 70/20/10 using seed 42.
- DenseNet121 produces a single pneumonia logit. Optional ImageNet initialization is requested with `--pretrained`.
- Train using image-level pneumonia labels and weighted binary cross-entropy. Bounding boxes are not training targets.
- Grad-CAM from the final DenseNet feature normalization layer produces a heatmap and one threshold-derived bounding rectangle.
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

The final run used a reproducible 1,000-image experiment, seed 42, two epochs, 128×128 inputs, batch size 8, and ImageNet-pretrained DenseNet121. `--pretrained` requests ImageNet weights and may need network access or a cached torchvision weight file; omit it for random initialization if weights cannot be obtained.

```bash
python -m src.train --data-root data/raw --max-images 1000 --seed 42 --epochs 2 --batch-size 8 --image-size 128 --pretrained
```

The selected IDs/split are recorded at `results/selection_manifest.csv`; the best validation-loss checkpoint is `checkpoints/best_model.pt`. Supported subset sizes up to 15,000 include 1000, 5000, 10000, and 15000. A request above the cap fails clearly.

## Evaluation command

```bash
python -m src.evaluate --data-root data/raw --max-images 1000 --seed 42 --checkpoint checkpoints/best_model.pt --results results
```

Evaluation writes `results/metrics.json`, `results/localization_metrics.json`, `results/confusion_matrix.svg`, `results/roc_curve.svg`, and PPT-ready paired original/Grad-CAM images in `results/examples/`. Metrics include accuracy, precision, sensitivity/recall, specificity, F1, AUROC, confusion matrix, localization mean IoU, and the percentage of examples with IoU ≥ 0.5. The box rule is one Grad-CAM box per image and maximum IoU against that image's ground-truth boxes.

## Streamlit demo

After installing requirements and training the checkpoint:

```bash
streamlit run app.py
```

Upload PNG, JPG, or DICOM to display pneumonia probability, binary prediction, Grad-CAM, and the predicted heatmap box. A checkpoint is required.

## Device selection

The code selects CUDA if available, then Apple MPS using `torch.backends.mps.is_available()`, then CPU.

## Actual final results and current limitations

Run date: 2026-10-08. We trained on 1,000 images selected from the local training set (225 positive, 775 negative), seed 42; split counts were train 700 (160 positive / 540 negative), validation 200 (38 / 162), and test 100 (27 / 73). The experiment used two epochs, 128×128 inputs, batch size 8, ImageNet-pretrained DenseNet121, AdamW at 1e-4, and MPS for training. The best validation-loss checkpoint was saved at epoch 2 (validation loss 0.7705). Evaluation ran on CPU in this environment.

Held-out classification results at probability threshold 0.5:

| Metric | Result |
|---|---:|
| Accuracy | 0.7500 |
| Precision | 0.5294 |
| Recall / sensitivity | 0.6667 |
| Specificity | 0.7808 |
| F1 | 0.5902 |
| AUROC | 0.8057 |
| Confusion matrix (actual rows [negative, positive], predicted columns [negative, positive]) | `[[57, 16], [9, 18]]` |

Localization on 27 pneumonia-positive test images with boxes used one threshold-derived Grad-CAM box per image and maximum IoU against that image's GT boxes. Primary classifier-conditioned mean IoU was **0.0839**, with **0%** of examples at IoU ≥ 0.5. Classifier-independent mean IoU was **0.0906**, also with **0%** at IoU ≥ 0.5. These results show moderate classification discrimination on this small split, but weak bounding-box localization.

Generated artifacts: `checkpoints/best_model.pt`; `results/metrics.json`; `results/localization_metrics.json`; `results/confusion_matrix.svg`; `results/roc_curve.svg`; `results/selection_manifest.csv`; and high-resolution PNGs under `results/examples/`. The results folder is ignored by Git to keep generated artifacts and dataset-related outputs out of the public repository.

Limitations: only 1,000 images and 100 held-out cases were used; the test split has 27 positives; localization is poor at the chosen Grad-CAM threshold; training ran on MPS while evaluation ran on CPU in this environment; Grad-CAM boxes are coarse explanations rather than validated lesion segmentations; and the reference paper leaves its CAM IoU interpretation ambiguous.

## PPT and report

[`docs/PPT_OUTLINE.md`](docs/PPT_OUTLINE.md) contains the requested 16-slide content and actual results. [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) contains the two-page report with the same run results and limitations.
