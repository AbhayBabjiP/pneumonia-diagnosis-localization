# Dataset preparation and evaluation policy

## Local data

Obtain the RSNA Pneumonia Detection Challenge dataset separately and place it under `data/raw/`. This project does not download, alter, or commit the source dataset. `data/raw/` is reserved for local data and should remain untracked.

## Reproducible subset

The default maximum is `max_dataset_images: 15000` and default random seed is `42` (see `configs/config.yaml`). The preparation interface accepts `--max_images`; supported experiment sizes include 1000, 5000, 10000, and 15000. Any request greater than 15000 raises an error. Sampling is stratified on image-level pneumonia target. The label rows are first aggregated by image, retaining every bounding box attached to that image; image IDs, never individual boxes, are sampled. The train/validation/test split (70/20/10 by default) happens after subset selection and is reproducible with the configured seed.

Run, after the RSNA labels CSV exists:

```bash
python -m src.data.selection --max_images 15000 --seed 42
```

The default labels path is `data/raw/stage_2_train_labels.csv`. The command reports selected count, positive/negative counts, positive percentage, and split counts based on the actual local file. It does not copy, delete, or rewrite dataset files. For a selected image, all of its ground-truth boxes remain available in the aggregated record.

## Evaluation

Classification is evaluated across the entire held-out test partition. Primary localization evaluation covers all pneumonia-positive test images with ground-truth boxes and is conditioned on the classifier decision: a false negative receives localization IoU 0. A secondary classifier-independent localization score should also be reported when practical, applying localization to every eligible positive regardless of the classifier output. Report image and box counts and the multi-box matching/aggregation rule. See the reference paper analysis for the source's unresolved evaluation ambiguity and the project protocol.
