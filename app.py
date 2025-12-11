# app.py
# Streamlit app integrating YOLO (damage detection) + Ripeness CNN classifier
# Put best.pt, ripeness_cnn.keras (or ripeness_cnn.h5) and class_labels.json in same folder.
# Run: streamlit run app.py

import os
import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ML libs
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_yolo_safe(model_path: str):
    if YOLO is None:
        return None, "ultralytics (YOLO) package not installed"
    if not os.path.exists(model_path):
        return None, f"YOLO model not found at: {model_path}"
    try:
        model = YOLO(model_path, task="detect")
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_cnn_safe(candidate_paths=("ripeness_cnn.keras", "ripeness_cnn.h5", "ripeness_checkpoint.keras")):
    for p in candidate_paths:
        if not p:
            continue
        p = p.strip()
        if not os.path.exists(p):
            continue
        try:
            model = tf.keras.models.load_model(p, compile=False)
            return model, p, None
        except Exception as e:
            last_err = e
    return None, None, "No CNN model found in candidate paths."

def load_labels(path="class_labels.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                labels = json.load(f)
            if isinstance(labels, list) and len(labels) > 0:
                return labels
        except Exception:
            pass
    # fallback
    return ["overripe", "ripe", "unripe"]

def yolo_names_list(yolo_model):
    """Return list of names in order index->name (works if model.names is dict or list)."""
    if yolo_model is None:
        return []
    names = getattr(yolo_model, "names", None)
    if names is None:
        return []
    if isinstance(names, dict):
        # sort keys to build a list
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, (list, tuple)):
        return list(names)
    # fallback
    return list(names)

def draw_boxes_on_pil(img_pil, detections, names_list, conf_threshold=0.35):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(255,0,255)]

    for det in detections:
        try:
            conf = float(det.conf)
        except Exception:
            continue
        if conf < conf_threshold:
            continue
        try:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        except Exception:
            continue
        cls = int(det.cls.item()) if hasattr(det, "cls") else 0
        label = names_list[cls] if cls < len(names_list) else f"class_{cls}"
        xmin, ymin, xmax, ymax = xyxy
        color = colors[cls % len(colors)]
        draw.rectangle([xmin,ymin,xmax,ymax], outline=color, width=2)
        text = f"{label} {conf:.2f}"
        # textbbox is robust across Pillow versions
        try:
            bbox = draw.textbbox((0,0), text, font=font)
            text_w = bbox[2]-bbox[0]; text_h = bbox[3]-bbox[1]
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([xmin, ymin-text_h-6, xmin+text_w+6, ymin], fill=color)
        draw.text((xmin+3, ymin-text_h-4), text, fill=(0,0,0), font=font)
    return img_pil

def model_has_internal_preprocessing(cnn_model):
    if cnn_model is None:
        return False
    names = [layer.name.lower() for layer in cnn_model.layers]
    flags = any(n in ("true_divide", "subtract", "rescaling") or "augmentation" in n or "data_augmentation" in n for n in names)
    return flags

def preprocess_for_cnn_auto(pil_img, cnn_model, target_size=(224,224)):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img)  # dtype=uint8, shape (H,W,3)
    if cnn_model is None:
        out = mobilenet_preprocess(arr.astype("float32"))
        return np.expand_dims(out, 0).astype("float32")
    if model_has_internal_preprocessing(cnn_model):
        return np.expand_dims(arr.astype("float32"), 0)
    else:
        out = mobilenet_preprocess(arr.astype("float32"))
        return np.expand_dims(out, 0).astype("float32")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fruit Quality — YOLO + Ripeness CNN", layout="wide")
st.title("🍋 Fruit Quality — Damage Detection (YOLO) + Ripeness (CNN)")

st.sidebar.header("Model files & settings")
yolo_path = st.sidebar.text_input("YOLO model path (best.pt)", "best.pt")
cnn_paths_txt = st.sidebar.text_input("CNN model filenames (comma)", "ripeness_cnn.keras, ripeness_cnn.h5, ripeness_checkpoint.keras")
cnn_candidates = [s.strip() for s in cnn_paths_txt.split(",") if s.strip()]
labels_path = st.sidebar.text_input("Ripeness labels JSON", "class_labels.json")

# confidence slider (important for webcam: you can lower it)
conf_threshold = st.sidebar.slider("YOLO confidence threshold", 0.05, 0.9, 0.35, step=0.01)

if st.sidebar.button("Reload models"):
    st.experimental_rerun()

with st.spinner("Loading models..."):
    yolo_model, yolo_err = load_yolo_safe(yolo_path)
    cnn_model, cnn_loaded_path, cnn_err = load_cnn_safe(tuple(cnn_candidates))
    ripeness_labels = load_labels(labels_path)
    yolo_names = yolo_names_list(yolo_model)

# sidebar status
if yolo_model is None:
    st.sidebar.warning(f"YOLO: {yolo_err}")
else:
    st.sidebar.success(f"YOLO loaded from {yolo_path}")

if cnn_model is None:
    msg = cnn_err if cnn_err else "No CNN loaded"
    st.sidebar.warning(f"CNN: {msg}")
else:
    st.sidebar.success(f"CNN loaded: {cnn_loaded_path}")
    if model_has_internal_preprocessing(cnn_model):
        st.sidebar.info("CNN: internal preprocessing detected → app will NOT call mobilenet preprocessing (avoid double-preprocess).")
    else:
        st.sidebar.info("CNN: no internal preprocessing → app will use MobileNetV2 preprocess_input before predict.")

# --- Input mode selection (Upload or Webcam) ---
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Input Mode")
    input_mode = st.radio("Choose Input Mode:", ("Upload Image", "Webcam"))
    st.write("Tip: lower YOLO confidence for webcam if detections miss small marks.")
    if st.button("Reload models (sidebar)"):
        st.experimental_rerun()

with col2:
    st.subheader("Input")

# capture the input image (either uploaded file or webcam frame)
uploaded = None
camera_image = None

if input_mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
else:  # Webcam
    st.info("Use your device camera to capture a photo. If you want to use a phone image, either upload the file or show the phone's screen to your laptop webcam.")
    camera_image = st.camera_input("Capture image from Webcam")

# Trigger inference button (same on either mode)
run_btn = st.button("🚀 Run inference")

# ---------- Inference helpers ----------
def run_inference_on_pil(image_pil):
    """Runs YOLO detection and CNN ripeness on a PIL.Image and returns (vis_image_pil, results_rows)."""
    detections = []
    if yolo_model is not None:
        try:
            results = yolo_model(np.array(image_pil), verbose=False)
            res0 = results[0]
            detections = res0.boxes
        except Exception as e:
            st.error(f"YOLO inference failed: {e}")
            detections = []

    vis = image_pil.copy()
    vis = draw_boxes_on_pil(vis, detections, yolo_names if yolo_names else ["class0"], conf_threshold)

    rows = []
    for det in detections:
        try:
            conf = float(det.conf)
        except Exception:
            continue
        if conf < conf_threshold:
            continue
        try:
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        except Exception:
            continue
        xmin, ymin = max(0, xyxy[0]), max(0, xyxy[1])
        xmax, ymax = min(image_pil.width, xyxy[2]), min(image_pil.height, xyxy[3])
        crop = image_pil.crop((xmin, ymin, xmax, ymax))

        ripeness_pred = "-"
        ripeness_conf = None
        if cnn_model is not None:
            try:
                x = preprocess_for_cnn_auto(crop, cnn_model, target_size=(224,224))
                pred = cnn_model.predict(x, verbose=0)[0]
                idx = int(np.argmax(pred))
                ripeness_pred = ripeness_labels[idx] if idx < len(ripeness_labels) else str(idx)
                ripeness_conf = float(pred[idx])
            except Exception as e:
                ripeness_pred = f"err:{str(e)[:20]}"

        classidx = int(det.cls.item()) if hasattr(det, "cls") else 0
        detected_label = yolo_names[classidx] if classidx < len(yolo_names) else str(classidx)
        rows.append({
            "detection_label": detected_label,
            "detection_conf": f"{conf:.2f}",
            "ripeness": ripeness_pred,
            "ripeness_conf": f"{ripeness_conf:.2f}" if ripeness_conf is not None else "-"
        })

    # fallback: if no boxes but CNN present, predict on full image
    if len(rows) == 0 and cnn_model is not None:
        try:
            x = preprocess_for_cnn_auto(image_pil, cnn_model, target_size=(224,224))
            pred = cnn_model.predict(x, verbose=0)[0]
            idx = int(np.argmax(pred))
            ripeness_pred = ripeness_labels[idx] if idx < len(ripeness_labels) else str(idx)
            ripeness_conf = float(pred[idx])
            rows.append({
                "detection_label": "full_image",
                "detection_conf": "-",
                "ripeness": ripeness_pred,
                "ripeness_conf": f"{ripeness_conf:.2f}"
            })
        except Exception:
            pass

    return vis, rows

# ---------- Main ----------
if run_btn:
    source_pil = None
    if input_mode == "Upload Image":
        if uploaded is None:
            st.warning("Please upload an image.")
            st.stop()
        try:
            img_bytes = uploaded.read()
            source_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read uploaded image: {e}")
            st.stop()
    else:  # webcam
        if camera_image is None:
            st.warning("Please capture a webcam photo first.")
            st.stop()
        try:
            img_bytes = camera_image.getvalue()
            source_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read camera image: {e}")
            st.stop()

    # run inference
    with st.spinner("Running inference..."):
        vis_img, rows = run_inference_on_pil(source_pil)

    # show detection image
    st.image(vis_img, caption="YOLO detections", use_container_width=True)

    # show table
    if rows:
        df = pd.DataFrame(rows)
        st.markdown("### Detections & Ripeness Predictions")
        st.dataframe(df, width=900)
        # Overall quality rule
        DAMAGE_KEYWORDS = ["rotten","damaged","bad"]
        is_damaged = any(any(k in str(lbl).lower() for k in DAMAGE_KEYWORDS) for lbl in df["detection_label"])
        st.markdown(f"### Overall quality: **{'BAD (damage detected)' if is_damaged else 'GOOD'}**")
    else:
        st.info("No detections and no CNN predictions available.")

st.markdown("---")
st.markdown("- Put best.pt, ripeness_cnn.keras (or .h5) and class_labels.json in same folder as app.py.")
st.markdown("- Install: pip install ultralytics tensorflow streamlit pillow pandas numpy opencv-python")
st.markdown("- Run: streamlit run app.py")
