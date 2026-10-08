"""Evaluate classification and weakly supervised Grad-CAM localization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.rsna import RsnaDataset, prepare_split
from src.device import select_device
from src.metrics import binary_metrics, box_iou, heatmap_box, save_confusion_svg, save_roc_svg, save_visual, write_json
from src.model import GradCAM, build_model


def gt_boxes_for_size(record, dicom_path, output_size):
    import pydicom
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    sx = output_size / float(ds.Columns); sy = output_size / float(ds.Rows)
    return [[float(b[k]) * scale for k, scale in zip(("x", "y", "width", "height"), (sx, sy, sx, sy))] for b in record["boxes"]]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--results", default="results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-visuals", type=int, default=12)
    args = parser.parse_args(argv)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
    root, records, selected, splits = prepare_split(args.data_root, args.max_images, args.seed)
    device = select_device(torch)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    size = checkpoint.get("image_size", 224)
    ds = RsnaDataset(root, splits["test"], records, size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    y=[]; probabilities=[]; ids=[]
    with torch.no_grad():
        for images, labels, batch_ids in loader:
            logits=model(images.to(device)).flatten()
            y.extend(labels.numpy().astype(int).tolist())
            probabilities.extend(logits.sigmoid().cpu().numpy().tolist())
            ids.extend(batch_ids)
    metrics=binary_metrics(y, probabilities)
    out=Path(args.results); out.mkdir(parents=True,exist_ok=True)
    save_confusion_svg(metrics["confusion_matrix"],out/"confusion_matrix.svg")
    save_roc_svg(y,probabilities,out/"roc_curve.svg")
    experiment={"selected_images":selected.selected_count,"positive_images":selected.positive_count,"negative_images":selected.negative_count,"seed":args.seed,"split_counts":{k:len(v) for k,v in splits.items()},"split_class_counts":{name:{"positive":sum(int(records[i]["target"]) for i in ids),"negative":sum(1-int(records[i]["target"]) for i in ids)} for name,ids in splits.items()},"device":str(device),"training_device":checkpoint.get("train_device","not-recorded"),"epochs_requested":checkpoint.get("epochs_requested"),"best_epoch":checkpoint.get("best_epoch"),"best_validation_loss":checkpoint.get("best_validation_loss"),"batch_size":checkpoint.get("batch_size"),"image_size":checkpoint.get("image_size"),"pretrained":checkpoint.get("pretrained"),"optimizer":checkpoint.get("optimizer"),"learning_rate":checkpoint.get("learning_rate"),"checkpoint":str(checkpoint_path)}
    write_json({"classification":metrics,"experiment":experiment},out/"metrics.json")
    cam_engine=GradCAM(model)
    localization={"conditioned":[],"independent":[]}
    visual_candidates=[]
    for image_id, label, probability in zip(ids,y,probabilities):
        if label != 1 or not records[image_id]["boxes"]: continue
        original=ds.image_ids.index(image_id)
        x,_,_=ds[original]
        tensor=x.unsqueeze(0).to(device)
        pred=probability>=0.5
        gt=gt_boxes_for_size(records[image_id],root/"stage_2_train_images"/f"{image_id}.dcm",size)
        if pred:
            cam=cam_engine(tensor,1); pred_box=heatmap_box(cam,(size,size))
            iou=max((box_iou(pred_box,b) for b in gt),default=0.0)
            localization["conditioned"].append(iou)
            category="positive_correctly_classified_and_localized" if iou >= 0.5 else "positive_incorrectly_localized"
        else:
            localization["conditioned"].append(0.0)
            with torch.enable_grad(): cam=cam_engine(tensor,1)
            pred_box=heatmap_box(cam,(size,size))
            category="positive_missed_by_classifier"
        with torch.enable_grad():
            cam_ind=cam_engine(tensor,1)
        independent_box=heatmap_box(cam_ind,(size,size))
        independent_iou=max((box_iou(independent_box,b) for b in gt),default=0.0)
        localization["independent"].append(independent_iou)
        from src.data.rsna import dicom_to_rgb
        visual_size=512; scale=visual_size/size
        image=dicom_to_rgb(root/"stage_2_train_images"/f"{image_id}.dcm",visual_size)
        visual_box=[coordinate*scale for coordinate in independent_box]
        visual_gt=[[coordinate*scale for coordinate in box] for box in gt]
        visual_candidates.append({"category":category,"image_id":image_id,"probability":probability,"iou":independent_iou,"image":image,"cam":cam_ind.detach().cpu(),"box":visual_box,"gt":visual_gt,"positive_prediction":bool(pred)})
    # Reserve visual slots for a true-positive classification, a weak localization,
    # and a negative image before filling remaining slots with other positives.
    positives=visual_candidates
    true_positives=[c for c in positives if c["positive_prediction"]]
    correctly_classified=max(true_positives,key=lambda c:c["iou"],default=None)
    incorrect_candidates=[c for c in true_positives if c is not correctly_classified and c["iou"]<0.5]
    incorrect_localized=min(incorrect_candidates,key=lambda c:c["iou"],default=None)
    negatives=[]
    for image_id, target, probability in zip(ids,y,probabilities):
        if target != 0: continue
        idx=ds.image_ids.index(image_id); x,_,_=ds[idx]
        with torch.enable_grad(): cam=cam_engine(x.unsqueeze(0).to(device),0)
        box=heatmap_box(cam,(size,size))
        from src.data.rsna import dicom_to_rgb
        visual_size=512; scale=visual_size/size
        image=dicom_to_rgb(root/"stage_2_train_images"/f"{image_id}.dcm",visual_size)
        negatives.append({"category":"negative_example","image_id":image_id,"probability":probability,"image":image,"cam":cam.detach().cpu(),"box":[coordinate*scale for coordinate in box],"gt":[]})
        if len(negatives)>=2: break
    chosen=[]
    if correctly_classified is not None:
        correctly_classified=dict(correctly_classified,category="positive_correctly_classified")
    if incorrect_localized is not None:
        incorrect_localized=dict(incorrect_localized,category="positive_incorrectly_localized")
    for item in (correctly_classified,incorrect_localized,negatives[0] if negatives else None):
        if item is not None and all(item["image_id"]!=prev["image_id"] for prev in chosen): chosen.append(item)
    for item in positives:
        if len(chosen)>=args.max_visuals: break
        if all(item["image_id"]!=prev["image_id"] for prev in chosen): chosen.append(item)
    for item in negatives[1:]:
        if len(chosen)>=args.max_visuals: break
        chosen.append(item)
    visuals=[]
    for item in chosen:
        path=out/"examples"/f"{item['category']}_{item['image_id']}.png"
        title=f"{item['category']}; p={item['probability']:.3f}"
        if "iou" in item: title+=f"; IoU={item['iou']:.3f}"
        save_visual(item["image"],item["cam"],item["box"],item["gt"],path,title=title)
        visuals.append({k:item[k] for k in ("category","image_id","probability") if k in item} | ({"iou":item["iou"]} if "iou" in item else {}) | {"path":str(path)})
    cam_engine.close()
    def loc_summary(values):
        arr=np.asarray(values,dtype=float)
        return {"count":len(arr),"mean_iou":float(arr.mean()) if len(arr) else None,"percent_iou_ge_0_5":float((arr>=0.5).mean()*100) if len(arr) else None}
    loc={"primary_classifier_conditioned":loc_summary(localization["conditioned"]),"classifier_independent":loc_summary(localization["independent"]),"box_rule":"For each positive test image, one Grad-CAM box; image IoU is the maximum IoU against any ground-truth box. Classifier false negatives receive 0 in the primary score.","examples":visuals}
    metrics["localization"]=loc
    write_json({"classification":{k:v for k,v in metrics.items() if k != "localization"},"localization":loc,"experiment":experiment},out/"metrics.json")
    write_json(loc,out/"localization_metrics.json")
    print(f"Classification: {metrics}")
    print(f"Localization: {loc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
