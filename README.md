# Pneumonia Diagnosis Detection and Localization

This project classifies frontal chest X-rays for pneumonia and provides weakly supervised Grad-CAM localization. Image-level pneumonia labels train the classifier. RSNA bounding boxes are held out for localization evaluation and are not classification training targets.

## Dataset

The RSNA Pneumonia Detection Challenge dataset must be supplied locally under `data/raw/`. The full labeled set used in the final experiment has 30,227 annotation rows for 26,684 unique image IDs: 6,012 positive and 20,672 negative images. There are 3,398 images with multiple annotation rows. The experiment groups all rows and boxes by image before splitting. Dataset files are not changed or committed.

The reproducible split uses seed 42 and 80/10/10 fractions:

| Split | Images | Positive | Negative |
|---|---:|---:|---:|
| Train | 21,347 | 4,783 | 16,564 |
| Validation | 2,668 | 626 | 2,042 |
| Test | 2,669 | 603 | 2,066 |

## Method and architecture

- DenseNet121 produces one pneumonia logit; the final experiment initializes it with ImageNet pretrained weights.
- Training uses image-level targets, class-weighted binary cross-entropy, and AdamW.
- Validation data selects the classification threshold from `0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85`, maximizing F1; ties are broken by sensitivity, then specificity. The chosen `0.70` threshold is locked before test evaluation.
- Grad-CAM uses the final DenseNet feature layer. Localization evaluation adaptively thresholds the heatmap, cleans the binary mask morphologically, finds connected components, and compares candidate regions with grouped RSNA boxes using IoU.
- Primary localization is classifier-conditioned: a classifier false negative scores IoU 0. Classifier-independent localization is also reported.
- See [the architecture and evaluation flow](docs/architecture.mmd), [the reference paper analysis](docs/reference_paper_analysis.md), and [the exact final experiment record](docs/final_experiment.md).

## Setup

From the repository root, with the RSNA data present under `data/raw/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch device selection uses CUDA if available, then Apple MPS, then CPU.

## Final 224×224 experiment

The final experiment used all 26,684 labeled images, DenseNet121 with ImageNet pretrained weights, 224×224 inputs, batch size 8, AdamW at `1e-4`, seed 42, and early stopping patience 3. It completed six epochs; the lowest validation loss was at epoch 3 (0.6725). Training ran on MPS. Exact commands and all test metrics are in [`docs/final_experiment.md`](docs/final_experiment.md) and [`docs/final_results.md`](docs/final_results.md).

```bash
python -m src.train --data-root data/raw --max-images 26684 --seed 42 --epochs 10 --patience 3 --batch-size 8 --image-size 224 --pretrained --checkpoint checkpoints/densenet121_224_ep10.pt
python -m src.evaluate --data-root data/raw --max-images 26684 --seed 42 --checkpoint checkpoints/densenet121_224_ep10.pt --results results/densenet121_224 --batch-size 8
```

The local final checkpoint can be used as `checkpoints/best_model.pt`. Checkpoints and generated result files are excluded from Git; obtain the checkpoint and result bundle separately. The public repository does not include the RSNA data.

## Final held-out results

At the validation-selected threshold 0.70, the 224×224 experiment achieved accuracy **82.73%**, precision **60.44%**, sensitivity **68.16%**, specificity **86.98%**, F1 **64.07%**, and AUROC **88.15%**. On 603 pneumonia-positive test images with boxes, classifier-conditioned mean/median IoU was **0.2051 / 0.1880**; classifier-independent mean/median IoU was **0.2307 / 0.2290**. Full comparison against the tuned 128×128 baseline is in [`docs/final_results.md`](docs/final_results.md).

## Streamlit demo

Install the dependencies and place the final checkpoint at `checkpoints/best_model.pt`, with final evaluation metrics at `results/metrics.json` or `results/densenet121_224/metrics.json`. From the repository root:

```bash
source .venv/bin/activate
streamlit run app.py
```

Upload a PNG, JPG, or DICOM chest X-ray. The app displays pneumonia probability, the thresholded class prediction, Grad-CAM overlay, and a heatmap-derived box. This box is an explanatory region, not a validated lesion segmentation.

## Limitations

The test split comes from the RSNA labeled training set and is not external validation. Final test accuracy is below 85%. The higher resolution improved AUROC and localization relative to the tuned baseline, but sensitivity and F1 were slightly lower. Grad-CAM localization remains approximate and is not clinically validated. See [`docs/final_results.md`](docs/final_results.md) for the measured comparison and [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) for the report summary.
