"""RSNA image loading and reproducible image-level dataset preparation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.selection import read_rsna_labels, select_stratified_images, split_image_ids


def find_dataset_root(configured: str | Path = "data/raw") -> Path:
    root = Path(configured)
    candidates = [root, *sorted(root.glob("*"))]
    for candidate in candidates:
        if (candidate / "stage_2_train_labels.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find stage_2_train_labels.csv under {root}. Expected the RSNA files in data/raw/ or one directory below it."
    )


def prepare_split(root: str | Path, max_images: int = 1000, seed: int = 42):
    root = find_dataset_root(root)
    labels = root / "stage_2_train_labels.csv"
    records = read_rsna_labels(labels)
    selected = select_stratified_images(records, max_images=max_images, seed=seed)
    splits = split_image_ids(selected.selected_ids, seed=seed)
    image_dir = root / "stage_2_train_images"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Found labels at {labels}, but missing DICOM image directory {image_dir}. "
            f"Cannot train: {len(records)} image IDs are labeled and no image files are available."
        )
    missing = [image_id for image_id in selected.selected_ids if not (image_dir / f"{image_id}.dcm").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {selected.selected_count} selected images lack matching DICOM files in {image_dir}. "
            f"First missing IDs: {', '.join(missing[:5])}"
        )
    return root, records, selected, splits


def dicom_to_rgb(path: str | Path, size: int = 224) -> Image.Image:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM loading requires pydicom. Install project requirements first.") from exc
    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    lo, hi = np.percentile(pixels, (0.5, 99.5))
    pixels = np.clip((pixels - lo) / max(float(hi - lo), 1e-6), 0, 1)
    image = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


class RsnaDataset(Dataset):
    def __init__(self, root, image_ids, records, size=224):
        self.root = Path(root)
        self.image_dir = self.root / "stage_2_train_images"
        self.image_ids = list(image_ids)
        self.records = records
        self.size = size
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = dicom_to_rgb(self.image_dir / f"{image_id}.dcm", self.size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        target = torch.tensor(float(self.records[image_id]["target"]), dtype=torch.float32)
        return tensor, target, image_id


def write_selection_manifest(root, selected, splits, records, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patientId", "target", "split", "boxes"])
        split_lookup = {image_id: name for name, ids in splits.items() for image_id in ids}
        for image_id in selected.selected_ids:
            writer.writerow([image_id, records[image_id]["target"], split_lookup[image_id], len(records[image_id]["boxes"])])
