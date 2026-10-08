"""Streamlit inference demo for the trained RSNA DenseNet121 checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw

from src.data.rsna import dicom_to_rgb
from src.device import select_device
from src.metrics import heatmap_box
from src.model import GradCAM, build_model


CHECKPOINT = Path("checkpoints/best_model.pt")


def prepare_image(image: Image.Image, size: int):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


@st.cache_resource
def load_model(checkpoint_path):
    device = select_device(torch)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device, checkpoint


st.set_page_config(page_title="Pneumonia Diagnosis and Localization", layout="wide")
st.title("Pneumonia Diagnosis and Localization")
st.caption("DenseNet121 image classifier with weakly supervised Grad-CAM localization")

if not CHECKPOINT.exists():
    st.error(f"Model checkpoint not found: {CHECKPOINT}. Train the classifier before launching the demo.")
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg", "dcm"])
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".dcm"):
            import pydicom
            ds = pydicom.dcmread(uploaded)
            pixels = ds.pixel_array.astype(np.float32)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixels = pixels.max() - pixels
            lo, hi = np.percentile(pixels, (0.5, 99.5))
            image = Image.fromarray((np.clip((pixels-lo)/max(float(hi-lo),1e-6),0,1)*255).astype(np.uint8)).convert("RGB")
        else:
            image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read uploaded image: {exc}")
        st.stop()
    model, device, checkpoint = load_model(str(CHECKPOINT))
    size = int(checkpoint.get("image_size", 224))
    tensor = prepare_image(image, size).to(device)
    with torch.no_grad():
        probability = float(model(tensor).sigmoid().item())
    prediction = probability >= 0.5
    st.metric("Pneumonia probability", f"{probability:.1%}")
    st.subheader("Pneumonia" if prediction else "No pneumonia")
    with torch.enable_grad():
        gradcam = GradCAM(model)
        cam = gradcam(tensor, 1 if prediction else 0)
        gradcam.close()
    box = heatmap_box(cam, image.size)
    overlay = image.convert("RGB").resize((size, size)).copy()
    heat = Image.fromarray(np.uint8(cam.detach().cpu().numpy()*255)).resize((size,size))
    heat_rgb = Image.new("RGB", (size,size), (255,0,0))
    overlay = Image.blend(overlay, Image.composite(heat_rgb, Image.new("RGB",(size,size)), heat), .4)
    draw = ImageDraw.Draw(overlay)
    x,y,w,h=box; draw.rectangle((x,y,x+w,y+h),outline=(0,255,0),width=3)
    left,right=st.columns(2)
    left.image(image,caption="Uploaded X-ray",use_container_width=True)
    right.image(overlay,caption="Grad-CAM with predicted box (green)",use_container_width=True)
    st.caption(f"Device: {device}. The predicted box is a heatmap-derived region, not a clinically validated segmentation.")
