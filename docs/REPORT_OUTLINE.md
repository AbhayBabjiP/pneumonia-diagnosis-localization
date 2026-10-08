# Two-page project report

## Pneumonia Diagnosis Detection and Localization

### Problem statement

This project aims to classify chest radiographs as pneumonia or no pneumonia and to provide a weakly supervised visual localization of regions contributing to the positive prediction. Classification and localization are evaluated separately. The localization model does not use RSNA boxes as training targets; boxes are reserved for evaluation.

### Dataset

The RSNA labels CSV contains 30,227 annotation rows for 26,684 unique patientIds. Grouping rows by patientId gives 6,012 pneumonia-positive and 20,672 negative images. There are 3,398 images with multiple annotation rows, so box annotations were grouped before image-level sampling. The local `stage_2_train_images/` directory contains 26,684 DICOM images matching the labels.

The final experiment used all 26,684 images (6,012 positive, 20,672 negative), seed 42, and an 80/10/10 split: train 21,347 (4,783/16,564), validation 2,668 (626/2,042), test 2,669 (603/2,066), positive/negative. The original dataset was not modified or committed.

### Proposed approach and implementation

The 128×128 baseline is ImageNet-pretrained DenseNet121 with class-weighted BCE, AdamW (learning rate 1e-4), and batch size 32; its best checkpoint was epoch 2 (validation loss 0.6925). The final 224×224 experiment used the same model, loss, and optimizer with batch size 8, MPS, up to 10 epochs, and patience 3. It early-stopped after six epochs; the best checkpoint was epoch 3 (validation loss 0.6725). Thresholds 0.30–0.85 were compared on validation, and 0.70 was selected by F1 for both runs. Grad-CAM boxes use adaptive thresholding, morphological cleanup, and connected components. Ground-truth boxes were not training labels.

Classification evaluation reports accuracy, precision, sensitivity/recall, specificity, F1, AUROC, and confusion matrix over the complete test split. Localization uses the maximum IoU over predicted components and all available ground-truth boxes. The primary score includes every box-annotated positive and assigns IoU 0 when the classifier misses; classifier-independent localization is reported separately.

### Results

On the 2,669-image held-out test set at threshold 0.70, the 128×128 baseline had accuracy 0.8202, precision 0.5834, sensitivity 0.7131, specificity 0.8514, F1 0.6418, and AUROC 0.8697. The 224×224 experiment had accuracy 0.8273, precision 0.6044, sensitivity 0.6816, specificity 0.8698, F1 0.6407, and AUROC 0.8815, with confusion matrix `[[1797, 269], [192, 411]]`. The higher resolution improved accuracy, precision, specificity, and AUROC modestly; sensitivity and F1 were slightly lower than the tuned baseline. Both accuracy results are below 85%.

Localization was measured on the same 603 pneumonia-positive test images with boxes. For the 128×128 baseline, conditioned mean/median IoU was 0.1038/0.0645; 9.95% reached 0.3 and 0.50% reached 0.5. Independent mean/median IoU was 0.1187/0.0775; 10.28% reached 0.3 and 0.50% reached 0.5. For the 224×224 run, conditioned mean/median IoU was 0.2051/0.1880; 34.00% reached 0.3 and 5.97% reached 0.5. Independent mean/median IoU was 0.2307/0.2290; 37.81% reached 0.3 and 6.47% reached 0.5. Localization improved substantially, though absolute IoU remains modest. Results are under `results/baseline_tuned/` and `results/densenet121_224/`.

### Limitations

This evaluation uses the RSNA training partition and does not establish external clinical performance. Test accuracy remains below 85%, and the 224×224 run's sensitivity/F1 did not exceed the tuned 128×128 result. Localization improved but Grad-CAM boxes are not validated segmentations. The reference paper does not resolve whether its reported CAM IoU is conditioned on classifier predictions.

### Conclusion

The six-epoch 224×224 run improved AUROC to 0.881 and classifier-independent mean localization IoU to 0.231, compared with 0.870 and 0.119 for the tuned 128×128 baseline. Its test accuracy was 82.7%, and sensitivity/F1 were 68.2%/0.641. Localization improved, but remains a weakly supervised, non-clinical estimate.
