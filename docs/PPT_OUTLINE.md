# Final presentation: exact slide content

**Final run:** ImageNet-pretrained DenseNet121, trained for two epochs on a reproducible 1,000-image subset (seed 42, 128×128 inputs, batch size 8) using MPS. Evaluation used a 100-image held-out split. Localization was weak; present that result candidly.

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
- Experiment subset: 1,000 images, 225 positive / 775 negative, seed 42.
- Split: train 700 (160/540), validation 200 (38/162), test 100 (27/73), positive/negative.

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
- Select up to 1,000 images with seed 42; split selected IDs 70/20/10.
- Train with binary cross-entropy and class weighting.
- Evaluate classification on every held-out test image.
- Evaluate IoU on positive test images with boxes; classifier false negatives score zero in the primary result.

## 10. Training

- Command: `.venv/bin/python -m src.train --data-root data/raw --max-images 1000 --seed 42 --epochs 2 --batch-size 8 --image-size 128 --pretrained`
- Architecture: ImageNet-pretrained DenseNet121; AdamW, learning rate 1e-4, class-weighted BCE.
- Device: MPS. Best checkpoint was epoch 2, validation loss 0.7705.
- Checkpoint: `checkpoints/best_model.pt`.

## 11. Classification results

- Held-out test set: 100 images; confusion matrix `[[57, 16], [9, 18]]`.
- Accuracy **0.7500**, precision **0.5294**, sensitivity **0.6667**, specificity **0.7808**.
- F1 **0.5902**, AUROC **0.8057**.
- These are this project's results, not the reference paper's reported metrics.

## 12. Localization results

- 27 pneumonia-positive test images with ground-truth boxes.
- Primary classifier-conditioned mean IoU **0.0839**; **0%** reached IoU ≥ 0.5.
- Classifier-independent mean IoU **0.0906**; **0%** reached IoU ≥ 0.5.
- A classifier false negative receives IoU 0 in the primary score. Localization performance is weak.

## 13. Visual examples

- Use `results/examples/positive_correctly_classified_88c25715-03f5-474e-9cee-5ad1f7beb4ce.png` (classifier positive, IoU 0.365).
- Use `results/examples/positive_incorrectly_localized_38ffaafc-11bb-4ca1-8242-dbb576fc8cc6.png` (IoU 0.000).
- Use `results/examples/negative_example_17e48dc0-50da-4085-b6bc-4627c165bd13.png` (negative ground truth, predicted probability 0.640; false positive).
- Green box is Grad-CAM prediction; red boxes are RSNA ground truth.

## 14. Live demo

- Launch with `streamlit run app.py`; checkpoint is available at `checkpoints/best_model.pt`.
- Upload PNG, JPG, or DICOM; show predicted probability, class, Grad-CAM overlay, and predicted box.
- Upload a PNG, JPG, or DICOM and show probability, predicted class, Grad-CAM overlay, and predicted region box.

## 15. Limitations

- The 1,000-image subset and 100-image test set are small; results may vary with seed and split.
- Localization is weak: mean IoU 0.0839, with no IoU ≥ 0.5 on the primary metric.
- Grad-CAM boxes are coarse explanations, not clinically validated lesion outlines.
- The reference paper's localization evaluation remains ambiguous.

## 16. Conclusion

- The reproducible DenseNet121 classifier completed training and held-out evaluation.
- Classification showed moderate discrimination on this small split; Grad-CAM box localization was poor.
- Further data and localization improvements are needed; results are not suitable for clinical use.
