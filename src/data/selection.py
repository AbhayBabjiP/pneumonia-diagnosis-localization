"""Reproducible image-level sampling for the local RSNA dataset."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

MAX_DATASET_IMAGES = 26_684


@dataclass(frozen=True)
class SelectionReport:
    selected_ids: tuple[str, ...]
    positive_count: int
    negative_count: int

    @property
    def selected_count(self) -> int:
        return len(self.selected_ids)

    @property
    def positive_percentage(self) -> float:
        return 100.0 * self.positive_count / self.selected_count if self.selected_count else 0.0


def aggregate_image_labels(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate box rows into one record per image before any sampling.

    Expected RSNA columns are patientId, Target, and optionally x/y/width/height.
    Multiple boxes are retained as a list. Target is image-level and must agree
    across rows for a patient; conflicting labels raise ValueError.
    """
    images: dict[str, dict[str, object]] = {}
    boxes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("patientId", row.get("image_id", ""))).strip()
        if not image_id:
            raise ValueError("Each row must contain a non-empty patientId/image_id")
        try:
            target = int(row["Target"] if "Target" in row else row["target"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid image-level target for {image_id}") from exc
        if target not in (0, 1):
            raise ValueError(f"Target for {image_id} must be 0 or 1, got {target}")
        if image_id not in images:
            images[image_id] = {"image_id": image_id, "target": target, "boxes": boxes[image_id]}
        elif images[image_id]["target"] != target:
            raise ValueError(f"Conflicting targets for image {image_id}")
        if target == 1 and all(k in row and row[k] not in (None, "") for k in ("x", "y", "width", "height")):
            boxes[image_id].append({k: row[k] for k in ("x", "y", "width", "height")})
    return images


def select_stratified_images(
    image_records: Mapping[str, Mapping[str, object]],
    max_images: int = MAX_DATASET_IMAGES,
    seed: int = 42,
) -> SelectionReport:
    """Select at most max_images distinct image IDs, stratified by binary target."""
    if max_images < 1:
        raise ValueError("max_images must be at least 1")
    if max_images > MAX_DATASET_IMAGES:
        raise ValueError(f"max_images={max_images} exceeds the project maximum of {MAX_DATASET_IMAGES}")
    by_class: dict[int, list[str]] = {0: [], 1: []}
    for image_id, record in image_records.items():
        target = int(record["target"])
        if target not in by_class:
            raise ValueError(f"Target for {image_id} must be 0 or 1")
        by_class[target].append(str(image_id))

    total = sum(map(len, by_class.values()))
    count = min(total, max_images)
    if count == total:
        chosen = by_class[0] + by_class[1]
    else:
        # Largest-remainder allocation preserves the observed class proportions.
        exact = {c: count * len(ids) / total for c, ids in by_class.items()}
        allocation = {c: min(len(by_class[c]), int(exact[c])) for c in (0, 1)}
        remainder = count - sum(allocation.values())
        order = sorted((0, 1), key=lambda c: (exact[c] - int(exact[c]), len(by_class[c])), reverse=True)
        for c in order:
            if remainder and allocation[c] < len(by_class[c]):
                allocation[c] += 1
                remainder -= 1
        rng = random.Random(seed)
        chosen = []
        for c in (0, 1):
            class_ids = sorted(by_class[c])
            rng.shuffle(class_ids)
            chosen.extend(class_ids[: allocation[c]])
    chosen = list(chosen)
    random.Random(seed).shuffle(chosen)
    positives = sum(int(image_records[i]["target"]) for i in chosen)
    return SelectionReport(tuple(chosen), positives, len(chosen) - positives)


def split_image_ids(
    selected_ids: Sequence[str], seed: int = 42,
    train_fraction: float = 0.80, validation_fraction: float = 0.10, test_fraction: float = 0.10,
) -> dict[str, tuple[str, ...]]:
    """Reproducibly split the already-selected subset into three partitions."""
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1")
    ids = list(selected_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("selected_ids contains duplicate image IDs")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_fraction)
    n_validation = int(n * validation_fraction)
    return {
        "train": tuple(ids[:n_train]),
        "validation": tuple(ids[n_train:n_train + n_validation]),
        "test": tuple(ids[n_train + n_validation:]),
    }


def read_rsna_labels(csv_path: str | Path) -> dict[str, dict[str, object]]:
    """Read the supplied stage-2 labels CSV and aggregate boxes per image."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return aggregate_image_labels(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a reproducible RSNA image subset")
    parser.add_argument("--max_images", type=int, default=MAX_DATASET_IMAGES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", type=Path, default=None, help="Labels CSV; auto-detected from data/raw when omitted")
    args = parser.parse_args(argv)
    labels_path = args.labels
    if labels_path is None:
        candidates = [Path("data/raw/stage_2_train_labels.csv"), *Path("data/raw").glob("*/stage_2_train_labels.csv")]
        labels_path = next((p for p in candidates if p.is_file()), candidates[0])
    records = read_rsna_labels(labels_path)
    report = select_stratified_images(records, args.max_images, args.seed)
    print(f"Selected images: {report.selected_count:,}")
    print(f"Positive: {report.positive_count:,}")
    print(f"Negative: {report.negative_count:,}")
    print(f"Positive percentage: {report.positive_percentage:.1f}%")
    splits = split_image_ids(report.selected_ids, args.seed)
    print("Split counts: " + ", ".join(f"{name}={len(ids):,}" for name, ids in splits.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
