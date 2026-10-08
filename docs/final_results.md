# Final results: 128×128 baseline vs 224×224 experiment

Both experiments use the same 26,684 image IDs, seed 42, 80/10/10 split, threshold selection grid, weighted BCE, and Grad-CAM connected-component localization evaluation. Each experiment selects its own threshold from validation by maximum F1; both selected **0.70**. Classification is scored over all 2,669 test images. Localization is scored over 603 positive test images with ground-truth boxes.

## Classification

| Metric | 128×128 baseline (2 epochs, batch 32) | 224×224 final (6 epochs completed, batch 8) |
|---|---:|---:|
| Accuracy | 0.8202 (82.02%) | 0.8273 (82.73%) |
| Precision | 0.5834 (58.34%) | 0.6044 (60.44%) |
| Recall / sensitivity | 0.7131 (71.31%) | 0.6816 (68.16%) |
| Specificity | 0.8514 (85.14%) | 0.8698 (86.98%) |
| F1 | 0.6418 (64.18%) | 0.6407 (64.07%) |
| AUROC | 0.8697 (86.97%) | 0.8815 (88.15%) |

The 224×224 run modestly improved accuracy, precision, specificity, and AUROC. Sensitivity and F1 were slightly lower than the tuned baseline. Neither run reached 85% accuracy.

## Localization

IoU rates are percentages of the 603 eligible positive test images. Classifier-conditioned localization assigns IoU 0 to a classifier false negative; classifier-independent localization runs Grad-CAM regardless of the classifier decision.

| Protocol / metric | 128×128 baseline | 224×224 final |
|---|---:|---:|
| Conditioned mean IoU | 0.1038 | 0.2051 |
| Conditioned median IoU | 0.0645 | 0.1880 |
| Conditioned IoU ≥ 0.3 | 9.95% | 34.00% |
| Conditioned IoU ≥ 0.5 | 0.50% | 5.97% |
| Independent mean IoU | 0.1187 | 0.2307 |
| Independent median IoU | 0.0775 | 0.2290 |
| Independent IoU ≥ 0.3 | 10.28% | 37.81% |
| Independent IoU ≥ 0.5 | 0.50% | 6.47% |

Localization improved substantially in the 224×224 run. The values remain modest and Grad-CAM boxes are approximate explanatory regions, not validated lesion segmentations.

## Artifacts and limitations

The final experiment configuration and commands are in [`final_experiment.md`](final_experiment.md). The final local checkpoint is `checkpoints/best_model.pt`; machine-generated metrics and examples are in `results/`. Dataset, checkpoints, and large generated results are intentionally excluded from GitHub. Evaluation uses a held-out split from the RSNA labeled training partition, not an external cohort.
