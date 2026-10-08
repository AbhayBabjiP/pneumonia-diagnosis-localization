# Reference paper analysis: Weakly Supervised Pneumonia Localization

## Source hierarchy and scope

- **Primary source:** `56.pdf`, the complete seven-page paper by Shih-Cheng (Mars) Huang, Medi Monam, and Emanuel Cortes. This controls our description of the reference methodology.
- **Supplementary source:** `56 (1).pdf`, the one-page poster. It is used only to identify source discrepancies; it does not override the paper.
- The authors' reported experiments are summarized as reported, not treated as verified results for our project. The RSNA data will be examined only after it is supplied under `data/raw/`.

## A. What the reference paper did

### Goal and methodology

The paper proposes a two-stage system: a supervised binary CNN classifier predicts pneumonia versus no pneumonia, then a weakly supervised Class Activation Map (CAM) localizes pneumonia regions without using bounding-box labels to train that localization component. The system first uses U-Net to estimate pixelwise lung confidence, creates a segmented image by multiplying the original image by the lung mask, and feeds both original and segmented images as two input channels. Localization is applied to images classified as pneumonia-positive.

### Dataset

The paper says the Kaggle RSNA Pneumonia Detection Competition data contain **28,989 images**: 8,964 pneumonia, 8,525 healthy, and 11,500 diseased/no-pneumonia. It says the diseased/no-pneumonia cases (including examples such as edema, bleeding, atelectasis/collapse, lung cancer, and post-treatment changes) were removed to simplify the binary experiment. The resulting binary experiment contains **17,489 images**: 8,964 pneumonia and 8,525 healthy. It reports a 70/20/10 train/validation/test split. The paper calls this approximately balanced (51.25%/48.74%).

Pneumonia images have ground-truth bounding boxes. The paper describes coordinates as the lower-left corner (X,Y), width, and height; it reports an average area near 50,000 pixels and average dimensions of roughly 300 by 400 pixels. These are paper-reported statistics, not counts or measurements from our eventual local copy.

### Preprocessing and architecture

The paper resizes 1024×1024 images to 128×128 to reduce training cost and normalizes each pixel location by subtracting its mean and dividing by its standard deviation. A U-Net produces lung-pixel confidence, which is used to form a lung-segmented image. The original and segmented views are both supplied to the classifier.

The selected classifier is described as a 10-convolution-layer CNN with zero padding and ReLU. CAM compatibility is obtained with a Global Average Pooling (GAP) layer followed by a single fully connected layer for two classes. The paper says it began with VGG16 but reduced the architecture after observing overfitting; it also reports selecting 3×3 filters to retain localization precision. The paper does not provide a complete layer-by-layer specification, channel counts, or enough detail to reproduce the exact network solely from its text.

### Training

The paper reports Adam, learning rate 0.0001, and 20 epochs for the CNN. It compares SGD, Adam, and Adagrad, and learning rates 0.001, 0.0001, and another value printed ambiguously as `0001`; it says 0.001 caused unstable weights and Adam at 0.0001 performed best among those trials. It reports a 70/20/10 split, but does not fully specify split seed, exact preprocessing statistics scope, batch size, augmentation, or all training implementation details. Do not infer these missing settings.

### Localization procedure

The paper also builds a supervised R-CNN benchmark trained with ground-truth location labels. Its description uses region proposals over the classifier's shared convolutional feature map, class prediction, and box-coordinate prediction.

For weakly supervised localization, the paper computes CAM by taking the final convolutional feature maps and weighting them by the fully connected weights for the pneumonia class, then sums across feature maps. It scales the result to a three-channel heatmap, applies depth-first search to group nonzero heatmap pixels into connected clusters, forms a bounding rectangle from each cluster's coordinate extrema, and keeps boxes within two standard deviations of the predictions. It gives no sufficiently precise heatmap thresholding rule in the method description to reconstruct this step exactly.

### Reported results

For classification, the paper reports training accuracies of 75.86% (logistic regression), 74.17% (SVM), 86.39% (random forest), and 93.07% (CNN); test accuracies are 73.02%, 58.18%, 83.00%, and 92.47%, respectively. Its confusion matrix lists 2,230 true positives, 210 false negatives, 110 false positives, and 1,823 true negatives.

For localization, the results table reports R-CNN IoU of 0.1859 on training and 0.1266 on test, while the weakly supervised CNN+CAM test IoU is 0.1508 (training shown as N/A). The paper says IoU is intersection over union of predicted and ground-truth boxes and describes the weakly supervised result as 0.0242 higher than the supervised test result.

### Limitations and future work stated by the paper

The paper says 128×128 compression loses information; full-resolution training could help if compute allows. It identifies false negatives, spine activation, difficulty with small boxes, and a fixed heatmap cutoff as issues, and suggests image rotation/zoom augmentation, more training data, transfer learning from related work, smaller filters for small boxes, and a dynamic cutoff. It reports that feeding CAM after correcting classifier decisions hypothetically yields IoU 0.379, while acknowledging that errors in classification can lower overall localization scores. The authors describe performance as below human-level labeling and propose broader medical-image use as a future possibility if improved.

## B. What is ambiguous or differs between the sources

The seven-page paper is the primary source for our implementation. The poster is supplementary. These discrepancies are recorded without an invented explanation:

| Topic | `56.pdf` (primary paper) | `56 (1).pdf` (supplementary poster) | Treatment here |
|---|---|---|---|
| Dataset size | Describes 28,989 images before excluding 11,500 diseased/no-pneumonia cases; resulting binary experiment has 17,489 images. | Describes the dataset as consisting of 17,489 images without mentioning the 28,989 pre-exclusion total in that statement. | Use the paper's distinction. 17,489 is not described here as the original total. |
| Classification test results | Gives test accuracy 73.02% LR, 58.18% SVM, 83.00% RF, 92.47% CNN. | Its displayed table lists only LR 73.02% and SVM 58.18% under test; RF and CNN entries are absent in the poster rendering. | Use the complete paper table as primary. |
| Localization table | Gives R-CNN train/test IoU 0.1859/0.1266 and CNN+CAM test IoU 0.1508 (train N/A). | Poster table layout visibly misaligns/misses values; it shows R-CNN and CNN+CAM labels with 0.1266 and 0.1508, and the poster prose says an increase of 0.0242. | Report paper table values, while flagging the paper's separate evaluation ambiguity below. |
| Localization interpretation | Method says localization runs on positively classified images. Discussion says CAM IoU would be 0.379 if all images were correctly classified and fed to CAM. The table reports 0.1508 without clearly stating whether it is conditional on classifier predictions or computed on another set. | Poster says localization is on positively predicted images and reports 0.1508 versus 0.1266. | The table's evaluation interpretation remains ambiguous in the primary paper; neither source resolves it. Do not choose an interpretation. |
| Classifier architecture wording | Describes a 10-convolution CNN, GAP, and one FC layer; the paper also discusses dual original/segmented input. | Poster calls the pooling layer “Global Average Max Pool,” an inconsistent/unclear phrase. | Follow the paper's GAP description; do not infer what the poster phrase intended. |

Other paper-level reproduction details are underspecified, including complete layer dimensions, exact split seed, batch size, and precise thresholding/heatmap construction. These are missing details, not grounds to fill in assumptions as if they were reported facts.

## C. What our project will do

### Dataset handling and subset policy

Our project reads the locally supplied RSNA files from `data/raw/` and does not download, modify, or commit the original dataset. The project supports a reproducible stratified image-level subset, currently capped at the 26,684 labeled training images available locally. It aggregates all bounding-box rows by image before sampling, samples image IDs by image-level pneumonia target with a fixed seed, then creates the train/validation/test split from that selected subset. The current final experiment uses an 80/10/10 split and seed 42. Dataset counts and class distribution must be computed from the actual files; paper-reported figures must not be substituted for local inspection.

### Evaluation protocol (defined before modeling)

1. Create the selected subset first, then split it into train, validation, and held-out test partitions using the configured split seed (default 42). Use one fixed selected subset and split across experiments unless configuration changes explicitly.
2. Evaluate classification on every image in the complete held-out test set. Report the confusion matrix and clearly named classification metrics.
3. Evaluate localization on pneumonia-positive test images with available ground-truth boxes. Aggregate all ground-truth boxes per image; never treat box rows as independent image samples. Compare predicted and ground-truth boxes using IoU and state the aggregation rule in the model/evaluation implementation before reporting results.
4. **Primary, classifier-conditioned localization:** run the normal pipeline, where localization is emitted only if the classifier predicts positive. Include all pneumonia-positive test images with boxes in the denominator; a classifier false negative receives localization IoU 0. This measures end-to-end behavior and prevents missed cases disappearing from evaluation.
5. **Secondary, classifier-independent localization:** where practical, run the localization head/CAM on every pneumonia-positive test image with boxes regardless of the classifier decision and report IoU separately. This isolates localization quality from classification gate errors. Clearly label this as a diagnostic analysis, not the end-to-end result.
6. Report the number of images and boxes included for each metric, how multiple predicted boxes are matched/aggregated against multiple ground-truth boxes, and missing-prediction handling. Do not compare numbers to the paper as directly equivalent unless its evaluation interpretation and matching details can be established.

## D. Why our evaluation protocol is clearer

The protocol fixes the evaluation population and denominator in advance, evaluates classification over the entire held-out test set, and defines the primary localization score over all eligible pneumonia-positive test images. A missed positive prediction is visible as zero end-to-end localization rather than being silently removed. The additional classifier-independent localization score separates the localization component's behavior from the classifier gate. This makes the two effects inspectable without claiming to resolve the paper's ambiguity.

## Proposed improvements (not part of the paper's method)

The following are project design choices, not claims about what the authors did: a reproducible image-level subset capped at the 26,684 locally available labeled images; fixed seed configuration; explicit image-level aggregation before sampling; an 80/10/10 held-out protocol with classifier-conditioned and classifier-independent localization; and actual local dataset audits. Model architecture or training changes beyond the paper should be proposed separately and identified as project improvements.
