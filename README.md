# Pneumonia Diagnosis Detection and Localization

**UE24CS352A – Machine Learning | Mini-Project**

**Team:** [Dheeraj P – PES2UG24CS342] · [Abhay Babji P – PES2UG24CS342]
**Section:** [F]
**Problem statement:** Weakly supervised pneumonia localization on chest X-rays (based on Huang, Monam & Cortes, *Weakly Supervised Pneumonia Localization*, CS229 2018).

---

## 1. Overview

This project classifies frontal chest X-rays as pneumonia / no pneumonia and highlights the suspected region using **weakly supervised Grad-CAM localization**.

- The classifier is trained with **image-level labels only**.
- RSNA bounding boxes are **never used for training**. They are held out and used only to evaluate localization quality (IoU).

### Results at a glance

| Task | Metric | Result |
|---|---|---:|
| Classification | Accuracy | 82.73% |
| Classification | AUROC | 88.15% |
| Classification | Sensitivity / Specificity | 68.16% / 86.98% |
| Localization (classifier-conditioned) | Mean IoU | 0.2051 |
| Localization (classifier-independent) | Mean IoU | 0.2307 |

Full details are in [Section 7](#7-results) and [`docs/final_results.md`](docs/final_results.md).

---

## 2. Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/AbhayBabjiP/pneumonia-diagnosis-localization.git
cd pneumonia-diagnosis-localization

# 2. Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Download the trained model (see Section 4) into checkpoints/best_model.pt

# 4. Launch the demo
streamlit run app.py
```

Device selection is automatic: **CUDA → Apple MPS → CPU**. Training on CPU is slow; the demo runs fine on CPU.

---

## 3. Dataset

**RSNA Pneumonia Detection Challenge** (Kaggle). The dataset is **not** included in this repository and must be placed under `data/raw/`.

Download (requires a Kaggle account and API token):

```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d data/raw/
```

Expected layout (confirm against `src/data/rsna.py`):

```text
data/raw/
├── stage_2_train_labels.csv
└── stage_2_train_images/
    └── *.dcm
```

### Dataset statistics

| Item | Count |
|---|---:|
| Annotation rows | 30,227 |
| Unique image IDs | 26,684 |
| Positive images | 6,012 |
| Negative images | 20,672 |
| Images with multiple boxes | 3,398 |

Annotation rows are **grouped by image** before splitting, so boxes from the same image never leak across splits.

### Train / validation / test split (seed 42, 80/10/10)

| Split | Images | Positive | Negative |
|---|---:|---:|---:|
| Train | 21,347 | 4,783 | 16,564 |
| Validation | 2,668 | 626 | 2,042 |
| Test | 2,669 | 603 | 2,066 |

---

## 4. Trained Model (Checkpoint)

Checkpoints are too large for GitHub and are kept outside the repo.

The demo needs two local files that are not in the repository:

| File | Location | Download |
|---|---|---|
| Trained model (DenseNet121, 224×224) | `checkpoints/best_model.pt` | [PASTE DRIVE / RELEASE LINK HERE] |
| Final evaluation metrics | `results/metrics.json` **or** `results/densenet121_224/metrics.json` | Same link as above |

Create the folders if they do not exist:

```bash
mkdir -p checkpoints results/densenet121_224
```

Then move the downloaded files into place. The demo will not start correctly without them.

**Skipping the download:** if you have the RSNA data, you can regenerate both files yourself.
Run the training command in [Section 6](#6-usage) to produce the checkpoint, copy it to `checkpoints/best_model.pt`, then run the evaluation command to produce `metrics.json`. Training takes a long time on CPU, so downloading is recommended.

---

## 5. Method

### Architecture and training setup

| Component | Choice |
|---|---|
| Backbone | DenseNet121 (ImageNet pretrained) |
| Input size | 224 × 224 |
| Supervision | Image-level labels |
| Loss | Class-weighted BCE with logits |
| Optimizer | AdamW, learning rate `1e-4` |
| Batch size | 8 |
| Epoch limit / early stopping | 10 epochs, patience 3 |
| Seed | 42 |
| Localization | Grad-CAM on the final DenseNet feature layer |

### Classification pipeline

```text
Chest X-ray → Preprocessing → 224×224 → DenseNet121
            → Pneumonia probability → Validation-selected threshold (0.70)
            → Pneumonia / No Pneumonia
```

### Localization pipeline

```text
DenseNet feature maps → Grad-CAM → Heatmap normalization
                      → Adaptive threshold → Morphological cleanup
                      → Connected components → Candidate boxes
                      → IoU vs. RSNA ground-truth boxes
```

### Threshold selection

The classification threshold is chosen **on the validation set only**, then applied unchanged to the test set.

- Candidates: `0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85`
- Rule: maximize F1, break ties by sensitivity, then by specificity.
- Selected threshold: **0.70**

### Localization evaluation protocols

- **Classifier-conditioned:** images the classifier misses (false negatives) receive IoU = 0.
- **Classifier-independent:** localization is scored regardless of the classifier's decision.

For each image, IoU is the maximum overlap between any predicted component and any ground-truth box.

More detail: [`docs/architecture.mmd`](docs/architecture.mmd) · [`docs/reference_paper_analysis.md`](docs/reference_paper_analysis.md) · [`docs/final_experiment.md`](docs/final_experiment.md)

---

## 6. Usage

### Train

```bash
python -m src.train \
  --data-root data/raw \
  --max-images 26684 \
  --seed 42 \
  --epochs 10 \
  --patience 3 \
  --batch-size 8 \
  --image-size 224 \
  --pretrained \
  --checkpoint checkpoints/densenet121_224_ep10.pt
```

The final run completed 6 of 10 epochs (early stopping), with the best validation loss of **0.6725** at **epoch 3**. Training used Apple MPS.

### Evaluate

```bash
python -m src.evaluate \
  --data-root data/raw \
  --max-images 26684 \
  --seed 42 \
  --checkpoint checkpoints/densenet121_224_ep10.pt \
  --results results/densenet121_224 \
  --batch-size 8
```

### Run tests

```bash
python -m pytest tests/
```

### Streamlit demo

**Prerequisites:** `checkpoints/best_model.pt` and a `metrics.json` file must be in place (see [Section 4](#4-trained-model-checkpoint)). The RSNA dataset is **not** needed to run the demo.

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Upload a **PNG, JPG/JPEG, or DICOM** chest X-ray. The app shows:

1. Pneumonia probability
2. Thresholded prediction (Pneumonia / No Pneumonia)
3. Grad-CAM overlay
4. Heatmap-derived localization box

> The displayed box is an explanatory region from Grad-CAM, **not** a clinically validated lesion segmentation.

---

## 7. Results

### Classification (held-out test set, threshold = 0.70)

| Metric | Result |
|---|---:|
| Accuracy | 82.73% |
| Precision | 60.44% |
| Sensitivity / Recall | 68.16% |
| Specificity | 86.98% |
| F1 score | 64.07% |
| AUROC | 88.15% |

Confusion matrix (rows = actual, columns = predicted; order: negative, positive):

```text
[[1797, 269],
 [ 192, 411]]
```

### Localization (603 positive test images with ground-truth boxes)

| Protocol | Mean IoU | Median IoU | IoU ≥ 0.3 | IoU ≥ 0.5 |
|---|---:|---:|---:|---:|
| Classifier-conditioned | 0.2051 | 0.1880 | 34.00% | 5.97% |
| Classifier-independent | 0.2307 | 0.2290 | 37.81% | 6.47% |

### Key finding: effect of resolution

Raising input resolution from **128×128 to 224×224** substantially improved localization:

- Classifier-conditioned mean IoU: **0.1038 → 0.2051**
- Images with IoU ≥ 0.3: **9.95% → 34.00%**

AUROC, accuracy, precision and specificity also improved. Sensitivity and F1 were slightly lower. See [`docs/final_results.md`](docs/final_results.md).

### Comparison with the reference paper

| | Reference paper (CS229 2018) | This project |
|---|---|---|
| Classifier | Custom 10-layer CNN, trained from scratch | DenseNet121, ImageNet pretrained |
| Input size | 128 × 128 | 224 × 224 |
| Localization | CAM + DFS clustering | Grad-CAM + adaptive threshold + connected components |
| Mean IoU | 0.1508 | 0.2051 (conditioned) |

The two projects use different data subsets and splits, so this is an indicative comparison rather than a controlled one.

---

## 8. Repository Structure

```text
pneumonia-diagnosis-localization/
├── configs/
├── docs/
│   ├── architecture.mmd
│   ├── final_experiment.md
│   ├── final_results.md
│   ├── reference_paper_analysis.md
│   ├── PPT_OUTLINE.md
│   └── REPORT_OUTLINE.md
├── src/
│   ├── train.py          # training loop, early stopping
│   ├── evaluate.py       # classification + localization evaluation
│   ├── model.py          # DenseNet121 + Grad-CAM
│   ├── metrics.py        # accuracy, F1, AUROC, IoU
│   ├── device.py         # CUDA / MPS / CPU selection
│   └── data/
│       ├── rsna.py       # RSNA loading and grouping
│       └── selection.py  # reproducible splits
├── tests/
├── app.py                # Streamlit demo
├── requirements.txt
├── .gitignore
└── README.md
```

### Files kept outside GitHub

```text
data/    checkpoints/    results/    *.dcm    *.zip
```

---

## 9. Limitations

- The test set comes from the RSNA labeled training partition. It is **not** external validation.
- Grad-CAM localization is approximate. Image-level supervision does not teach the model exact lesion boundaries.
- Only about 6% of test images reach IoU ≥ 0.5.
- Sensitivity (68.16%) is modest, so some pneumonia cases are missed.
- This is an academic prototype and **must not** be used for clinical diagnosis.

### Possible improvements

- Data augmentation (rotation, zoom) and higher input resolution
- Dynamic per-image heatmap thresholds
- Chest X-ray–specific pretraining (e.g. CheXNet-style weights)
- External validation on a different dataset

---

## 10. Contributions

| Member | Contribution |
|---|---|
| [Member 1] | [e.g. data pipeline, model training, checkpoint] |
| [Member 2] | [e.g. evaluation, Streamlit demo, README, write-up] |

---

## 11. Conclusion

This project demonstrates weakly supervised pneumonia detection and approximate localization from chest X-rays using DenseNet121 and Grad-CAM. The final 224×224 experiment reached **82.73% accuracy** and **88.15% AUROC**, with a classifier-conditioned mean localization IoU of **0.2051**. Higher resolution substantially improved localization, while the remaining gap shows how hard it is to recover precise lesion regions from image-level labels alone.

## References

1. S.-C. Huang, M. Monam, E. Cortes. *Weakly Supervised Pneumonia Localization.* CS229 Project Report, Stanford, 2018.
2. B. Zhou et al. *Learning Deep Features for Discriminative Localization.* CVPR 2016.
3. R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
4. G. Huang et al. *Densely Connected Convolutional Networks.* CVPR 2017.
5. RSNA Pneumonia Detection Challenge, Kaggle.
