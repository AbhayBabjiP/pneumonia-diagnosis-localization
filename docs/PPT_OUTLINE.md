# Final presentation: exact slide content

**Final run:** ImageNet-pretrained DenseNet121, trained for two epochs on all 26,684 labeled images (seed 42, 128×128 inputs, batch size 32) using MPS. The 80/10/10 split has 2,669 held-out test images. Validation selected a fixed probability threshold of 0.8464. Accuracy was 83.5%, below 85%; localization remains weak.

## 1. Title

**Pneumonia Diagnosis Detection and Localization**  
RSNA Pneumonia Detection Challenge · DenseNet121 and Grad-CAM  
Presenter: [Name] · [Course / Institution] · [Date]

## 2. Problem statement

- Detect pneumonia from a frontal chest X-ray.
- Highlight image regions that contribute to the model prediction.
- Evaluate classification and localization as separate tasks.

## 3. Motivation

- Chest X-rays are widely used to assess pneumonia.
- A prediction is easier to inspect when accompanied by a visual explanation.
- Weakly supervised localization avoids using boxes as training targets.

## 4. Dataset

- Source: RSNA Pneumonia Detection Challenge training labels.
- Local training set: 26,684 DICOM files and 30,227 annotation rows across 26,684 patientIds.
- Grouped targets: 6,012 positive and 20,672 negative images; 3,398 images have multiple annotations.
- Experiment: all 26,684 labeled images, 6,012 positive / 20,672 negative, seed 42.
- Split: train 21,347 (4,783/16,564), validation 2,668 (626/2,042), test 2,669 (603/2,066), positive/negative.

## 5. Existing paper / method

- Huang, Monam, and Cortes, “Weakly Supervised Pneumonia Localization.”
- Paper method: binary CNN classifier followed by CAM; U-Net lung segmentation is used as additional input.
- The paper reports a 10-convolution CNN, GAP, one fully connected layer, and CAM heatmap clustering.
- Paper-reported values are historical reference results, not this project's results.

## 6. Limitations of paper

- 128×128 compression loses image information.
- Paper leaves details of architecture and evaluation protocol underspecified.
- Its table reports CAM IoU 0.1508, while discussion states a hypothetical 0.379 when classifier decisions are corrected; the evaluation interpretation is ambiguous.
- Paper notes missed positives, spine activation, small boxes, and fixed heatmap thresholds as issues.

## 7. Proposed improvement

- Use DenseNet121 transfer learning for the image-level classifier.
- Keep RSNA boxes out of classifier training targets; use them only for localization evaluation.
- Report classifier-conditioned and classifier-independent Grad-CAM localization separately.
- Use a reproducible, stratified image-level subset and retain all boxes for each selected image.

## 8. Architecture

- Input: DICOM X-ray, converted to normalized RGB tensor.
- Classifier: DenseNet121 with one binary logit.
- Explanation: Grad-CAM from the final DenseNet feature normalization layer.
- Localization: threshold Grad-CAM to produce one rectangular region.
- Demo: Streamlit upload, probability, class, overlay, and region box.

## 9. Methodology

- Aggregate all annotation rows by patientId.
- Select image IDs with seed 42; split 80/10/10 after image-level grouping.
- Train with binary cross-entropy and class weighting.
- Evaluate classification on every held-out test image.
- Evaluate IoU on positive test images with boxes; classifier false negatives score zero in the primary result.

## 10. Training

- Preserved baseline command: `.venv/bin/python -m src.train --data-root data/raw --max-images 26684 --seed 42 --epochs 2 --batch-size 32 --image-size 128 --pretrained`
- Architecture: ImageNet-pretrained DenseNet121; AdamW, learning rate 1e-4, class-weighted BCE.
- Baseline best checkpoint was epoch 2, validation loss 0.6925.
- 224×224 experiment: MPS, 6 epochs (early stopping after 3 non-improving epochs), best checkpoint at epoch 3 (validation loss 0.6725).
- Checkpoint: `checkpoints/best_model.pt`.

## 11. Classification results

- Validation threshold sweep selected **0.70** by F1 (validation F1 **0.6662**, sensitivity **0.6949**); held fixed for test.
- Held-out test set: 2,669 images; confusion matrix `[[1797, 269], [192, 411]]`.
- Accuracy **0.8273**, precision **0.6044**, sensitivity **0.6816**, specificity **0.8698**.
- F1 **0.6407**, AUROC **0.8815**. Accuracy, precision, specificity, and AUROC modestly improved over the tuned 128×128 baseline; sensitivity and F1 were similar/slightly lower.
- These are this project's results, not the reference paper's reported metrics.

## 12. Localization results

- 603 pneumonia-positive test images with ground-truth boxes.
- Adaptive threshold + morphological cleanup + connected components; IoU takes max across predicted components and GT boxes.
- Primary mean/median IoU **0.2051 / 0.1880**; **34.00%** reached 0.3 and **5.97%** reached 0.5.
- Independent mean/median IoU **0.2307 / 0.2290**; **37.81%** reached 0.3 and **6.47%** reached 0.5.
- Localization improved over the 128×128 tuned baseline (primary mean IoU 0.1038; independent 0.1187), though it remains imperfect.
- A classifier false negative receives IoU 0 in the primary score. Localization performance is weak.

## 13. Visual examples

- Use `results/examples/positive_correctly_classified_cb4a1fca-136c-4577-bcef-d934c9f14f2c.png` (classifier positive, IoU 0.639).
- Use `results/examples/positive_incorrectly_localized_e09dbb79-cdc9-44a1-9547-6f5969a170dc.png` (IoU 0.000).
- Use `results/examples/negative_example_68a43ce5-2021-4c3c-8e62-11a93d132ff5.png` (negative ground truth, predicted probability 0.316).
- Green box is Grad-CAM prediction; red boxes are RSNA ground truth.

## 14. Live demo

- Launch with `streamlit run app.py`; checkpoint is available at `checkpoints/best_model.pt`.
- Upload PNG, JPG, or DICOM; show predicted probability, class, Grad-CAM overlay, and predicted box.
- Upload a PNG, JPG, or DICOM and show probability, predicted class, Grad-CAM overlay, and predicted region box.

## 15. Limitations

- 224×224 test accuracy was 82.7%, below 85%; sensitivity was 68.2%.
- Localization improved: primary mean IoU 0.2051; 5.97% reached IoU ≥ 0.5.
- This is an evaluation on the RSNA training partition, not external clinical validation.
- Grad-CAM boxes are coarse explanations, not clinically validated lesion outlines.
- The reference paper's localization evaluation remains ambiguous.

## 16. Conclusion

- The 224×224 six-epoch experiment improved AUROC and Grad-CAM localization over baseline.
- Accuracy remained below 85%, and localization still needs improvement and external validation.
- Further data and localization improvements are needed; results are not suitable for clinical use.
