"""Train a reproducible DenseNet121 pneumonia classifier."""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".torch_cache"))
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.rsna import RsnaDataset, prepare_split, write_selection_manifest
from src.device import select_device
from src.model import build_model


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, optimizer, device, training):
    model.train(training)
    loss_sum = 0.0; count = 0
    with torch.set_grad_enabled(training):
        for images, labels, _ids in loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels); count += len(labels)
    return loss_sum / max(count, 1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without validation-loss improvement")
    parser.add_argument("--pretrained", action="store_true", help="Initialize from ImageNet weights (requires cached/downloadable weights)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    root, records, selected, splits = prepare_split(args.data_root, args.max_images, args.seed)
    write_selection_manifest(root, selected, splits, records, "results/selection_manifest.csv")
    print(f"Selected images: {selected.selected_count}; positive={selected.positive_count}; negative={selected.negative_count}; seed={args.seed}")
    print("Split sizes: " + ", ".join(f"{key}={len(value)}" for key, value in splits.items()))
    device = select_device(torch)
    print(f"Device: {device}; DenseNet121; pretrained={args.pretrained}; image_size={args.image_size}")
    datasets = {name: RsnaDataset(root, ids, records, args.image_size) for name, ids in splits.items()}
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", num_workers=args.workers) for name, ds in datasets.items()}
    model = build_model(pretrained=args.pretrained).to(device)
    positive = sum(int(records[i]["target"]) for i in splits["train"])
    negative = len(splits["train"]) - positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / max(positive, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = float("inf"); best_epoch = 0; stale_epochs = 0; checkpoint = Path(args.checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], loss_fn, optimizer, device, True)
        val_loss = run_epoch(model, loaders["validation"], loss_fn, optimizer, device, False)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed_min={(time.time()-started)/60:.1f}", flush=True)
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "model": "DenseNet121", "num_classes": 1,
                "pretrained": args.pretrained, "image_size": args.image_size,
                "seed": args.seed, "max_images": args.max_images,
                "selected_images": selected.selected_count,
                "positive_images": selected.positive_count,
                "negative_images": selected.negative_count,
                "train_device": str(device),
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "best_epoch": best_epoch,
                "best_validation_loss": best,
                "patience": args.patience,
            }, checkpoint)
            print(f"Saved best checkpoint: {checkpoint}", flush=True)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation-loss improvement.", flush=True)
                break
    print(f"Training complete in {(time.time()-started)/60:.1f} minutes; best checkpoint={checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
