# Two-page project report

## Pneumonia Diagnosis Detection and Localization

### Problem statement

This project aims to classify chest radiographs as pneumonia or no pneumonia and to provide a weakly supervised visual localization of regions contributing to the positive prediction. Classification and localization are evaluated separately. The localization model does not use RSNA boxes as training targets; boxes are reserved for evaluation.

### Dataset

The RSNA labels CSV contains 30,227 annotation rows for 26,684 unique patientIds. Grouping rows by patientId gives 6,012 pneumonia-positive and 20,672 negative images. There are 3,398 images with multiple annotation rows, so box annotations were grouped before image-level sampling. The local `stage_2_train_images/` directory contains 26,684 DICOM images matching the labels.

The final experiment used all 26,684 images (6,012 positive, 20,672 negative), seed 42, and an 80/10/10 split: train 21,347 (4,783/16,564), validation 2,668 (626/2,042), test 2,669 (603/2,066), positive/negative. The original dataset was not modified or committed.

### Proposed approach and implementation

The preserved baseline is ImageNet-pretrained DenseNet121 with class-weighted BCE, AdamW (learning rate 1e-4), batch size 32, and 128×128 inputs. The best checkpoint was epoch 2 (validation loss 0.6925). On validation, thresholds 0.30–0.85 were compared; 0.70 maximized F1 (0.6517) and was locked before test evaluation. Grad-CAM boxes now use adaptive thresholding, morphological cleanup, and connected components. Ground-truth boxes were not training labels. A separate 224×224, 10-epoch-cap run was attempted, but stopped before completing an epoch because the available runtime selected CPU.

Classification evaluation reports accuracy, precision, sensitivity/recall, specificity, F1, AUROC, and confusion matrix over the complete test split. Localization uses the maximum IoU over predicted components and all available ground-truth boxes. The primary score includes every box-annotated positive and assigns IoU 0 when the classifier misses; classifier-independent localization is reported separately.

### Results

On the 2,669-image held-out test set at threshold 0.70, accuracy was 0.8202, precision 0.5834, sensitivity 0.7131, specificity 0.8514, F1 0.6418, and AUROC 0.8697. The confusion matrix was `[[1759, 307], [173, 430]]` (actual negative/positive rows, predicted negative/positive columns). This operating point improves sensitivity and F1 over the previous 0.8464 threshold; accuracy is below 85%.

Localization was measured on 603 pneumonia-positive test images with boxes. The classifier-conditioned mean/median IoU was 0.1038/0.0645; 9.95% reached 0.3 and 0.50% reached 0.5. Classifier-independent mean/median IoU was 0.1187/0.0775; 10.28% reached 0.3 and 0.50% reached 0.5. Thus, adaptive component extraction has not improved mean IoU versus the previous baseline. The primary score assigns zero to classifier false negatives. Results are under `results/baseline_tuned/`; the baseline checkpoint remains `checkpoints/best_model.pt`.

### Limitations

This evaluation uses the RSNA training partition and does not establish external clinical performance. Test accuracy remains below 85%. Localization is weak and Grad-CAM boxes are not validated segmentations. The 224×224 longer training experiment remains incomplete because only CPU was available in the execution environment. The reference paper does not resolve whether its reported CAM IoU is conditioned on classifier predictions.

### Conclusion

Threshold optimization increased sensitivity to 71.3% and F1 to 0.642 at the cost of lower accuracy and specificity. AUROC remained 0.870. Multi-component Grad-CAM evaluation produced low IoU, so localization needs further work. A longer, higher-resolution training result is not yet available.
