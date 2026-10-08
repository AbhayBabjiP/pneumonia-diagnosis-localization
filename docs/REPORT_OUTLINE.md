# Two-page project report

## Pneumonia Diagnosis Detection and Localization

### Problem statement

This project aims to classify chest radiographs as pneumonia or no pneumonia and to provide a weakly supervised visual localization of regions contributing to the positive prediction. Classification and localization are evaluated separately. The localization model does not use RSNA boxes as training targets; boxes are reserved for evaluation.

### Dataset

The RSNA labels CSV contains 30,227 annotation rows for 26,684 unique patientIds. Grouping rows by patientId gives 6,012 pneumonia-positive and 20,672 negative images. There are 3,398 images with multiple annotation rows, so box annotations were grouped before image-level sampling. The local `stage_2_train_images/` directory contains 26,684 DICOM images matching the labels.

The final experiment used all 26,684 images (6,012 positive, 20,672 negative), seed 42, and an 80/10/10 split: train 21,347 (4,783/16,564), validation 2,668 (626/2,042), test 2,669 (603/2,066), positive/negative. The original dataset was not modified or committed.

### Proposed approach and implementation

The classifier is ImageNet-pretrained DenseNet121 with a one-logit output and class-weighted binary cross-entropy. It was trained for two epochs with AdamW (learning rate 1e-4), batch size 32, 128×128 inputs, and MPS. The best validation-loss checkpoint was epoch 2 (loss 0.6925). A decision threshold of 0.8464 was selected to maximize validation accuracy and then applied unchanged to test evaluation. Grad-CAM is computed from the final DenseNet feature block and converted into one threshold-derived rectangular prediction. Ground-truth boxes were not used as training labels.

Classification evaluation reports accuracy, precision, sensitivity/recall, specificity, F1, AUROC, and confusion matrix over the complete test split. Localization evaluation uses the maximum IoU between the single predicted region and each image's available ground-truth boxes. The primary localization score includes every box-annotated pneumonia-positive test image and assigns IoU 0 when the classifier misses the positive; a separate classifier-independent score measures the localization component when practical.

### Results

On the 2,669-image held-out test set, accuracy was 0.8351, precision 0.6945, sensitivity 0.4826, specificity 0.9380, F1 0.5695, and AUROC 0.8697. The confusion matrix was `[[1938, 128], [312, 291]]` (actual negative/positive rows, predicted negative/positive columns). Test accuracy remained below the requested 85% target; the higher threshold trades sensitivity for specificity.

Localization was measured on 603 pneumonia-positive test images with boxes. The primary classifier-conditioned mean IoU was 0.1174, and 1.0% reached IoU ≥ 0.5. The classifier-independent mean IoU was 0.1820, and 1.3% reached IoU ≥ 0.5. The primary score assigns zero to classifier false negatives. Metrics, SVG plots, selection manifest, and PNG examples are under `results/`; the checkpoint is `checkpoints/best_model.pt`. Paper results are not substituted for project results.

### Limitations

This evaluation uses the RSNA training partition and does not establish external clinical performance. The fixed 2,669-image test set is larger than the earlier run, but test accuracy remained below 85%; sensitivity at the validation-selected threshold is low. Grad-CAM localization is weak and a threshold-derived box is not a validated segmentation. The reference paper also does not resolve whether its reported CAM IoU is conditioned on classifier predictions.

### Conclusion

This run completed reproducible DenseNet121 training and Grad-CAM evaluation and produced a Streamlit inference interface. Test accuracy was 83.5%, below the requested 85%; AUROC was 0.870, while sensitivity and localization remained weak at the selected operating point. Further validation and localization improvements are needed before any clinical interpretation.
