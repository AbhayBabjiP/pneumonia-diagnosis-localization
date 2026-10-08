# Two-page project report

## Pneumonia Diagnosis Detection and Localization

### Problem statement

This project aims to classify chest radiographs as pneumonia or no pneumonia and to provide a weakly supervised visual localization of regions contributing to the positive prediction. Classification and localization are evaluated separately. The localization model does not use RSNA boxes as training targets; boxes are reserved for evaluation.

### Dataset

The RSNA labels CSV contains 30,227 annotation rows for 26,684 unique patientIds. Grouping rows by patientId gives 6,012 pneumonia-positive and 20,672 negative images. There are 3,398 images with multiple annotation rows, so box annotations were grouped before image-level sampling. The local `stage_2_train_images/` directory contains 26,684 DICOM images matching the labels.

The experiment used 1,000 images (225 positive, 775 negative), seed 42, stratified image-level selection, and a reproducible 70/20/10 split: train 700 (160/540), validation 200 (38/162), test 100 (27/73), positive/negative. The original dataset was not modified or committed.

### Proposed approach and implementation

The classifier is ImageNet-pretrained DenseNet121 with a one-logit output and class-weighted binary cross-entropy. It was trained for two epochs with AdamW (learning rate 1e-4), batch size 8, 128×128 inputs, and MPS. The best validation-loss checkpoint was epoch 2 (loss 0.7705). Grad-CAM is computed from the final DenseNet feature block and converted into one threshold-derived rectangular prediction. Ground-truth boxes were not used as training labels.

Classification evaluation reports accuracy, precision, sensitivity/recall, specificity, F1, AUROC, and confusion matrix over the complete test split. Localization evaluation uses the maximum IoU between the single predicted region and each image's available ground-truth boxes. The primary localization score includes every box-annotated pneumonia-positive test image and assigns IoU 0 when the classifier misses the positive; a separate classifier-independent score measures the localization component when practical.

### Results

On the 100-image held-out test set, accuracy was 0.7500, precision 0.5294, sensitivity 0.6667, specificity 0.7808, F1 0.5902, and AUROC 0.8057. The confusion matrix was `[[57, 16], [9, 18]]` (actual negative/positive rows, predicted negative/positive columns).

Localization was measured on 27 pneumonia-positive test images with boxes. The primary classifier-conditioned mean IoU was 0.0839, and 0% reached IoU ≥ 0.5. The classifier-independent mean IoU was 0.0906, also with 0% at IoU ≥ 0.5. The primary score assigns zero to classifier false negatives. Metrics, SVG plots, selection manifest, and PNG examples are under `results/`; the checkpoint is `checkpoints/best_model.pt`. Paper results are not substituted for project results.

### Limitations

The 1,000-image subset and 100-image test set are small and cannot establish clinical performance. Classification sensitivity is moderate in this run, and Grad-CAM localization is weak. Grad-CAM is a coarse explanation and a threshold-derived box is not a validated segmentation. The reference paper also does not resolve whether its reported CAM IoU is conditioned on classifier predictions.

### Conclusion

This run completed reproducible DenseNet121 training and Grad-CAM evaluation and produced a Streamlit inference interface. Classification showed moderate discrimination on the small held-out split; box localization did not reach IoU 0.5. Further validation and localization improvements are needed before any clinical interpretation.
