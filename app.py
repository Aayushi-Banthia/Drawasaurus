import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import models, transforms

APP_DIR = Path(__file__).parent
WEIGHTS_DIR = APP_DIR / "weights"
CLASSES_FILE = APP_DIR / "classes.txt"
IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="AI Draw Battle", page_icon="🎨", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#fff0f5,#e8fbfa); margin-bottom: 1rem;}
.hero h1 {margin:0; color:#25253b;}
.hero p {margin:.4rem 0 0; color:#56566a;}
[data-testid="stMetric"] {background:#fbfbff; border:1px solid #ececf5; padding:12px; border-radius:12px;}
</style>
<div class="hero">
<h1>🎨 AI Draw Battle</h1>
<p>Draw the target. Can you convince ResNet, VGG, and MobileNet?</p>
</div>
""", unsafe_allow_html=True)

def read_classes():
    if CLASSES_FILE.exists():
        return [x.strip() for x in CLASSES_FILE.read_text().splitlines() if x.strip()]
    return ["apple", "cat", "airplane", "fish", "house", "tree", "star", "bicycle"]

CLASSES = read_classes()
MODEL_SPECS = {
    "ResNet": ("resnet18", "resnet18.pt"),
    "VGG": ("vgg11", "vgg11.pt"),
    "MobileNet": ("mobilenet_v3_small", "mobilenet_v3_small.pt"),
}

@st.cache_resource
def load_judges(class_names):
    loaded = {}
    missing = []
    for label, (arch, filename) in MODEL_SPECS.items():
        path = WEIGHTS_DIR / filename
        if not path.exists():
            missing.append(label)
            continue
        if arch == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        elif arch == "vgg11":
            model = models.vgg11(weights=None)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(class_names))
        else:
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE).eval()
        loaded[label] = model
    return loaded, missing

def preprocess_canvas(canvas_rgba):
    # Canvas is white with dark ink; QuickDraw bitmap drawings are normalized to dark background/light strokes.
    image = Image.fromarray(canvas_rgba.astype("uint8"), "RGBA").convert("RGB")
    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray)
    # Remove near-white canvas and invert to dark background with bright strokes.
    ink = 255 - arr
    # Binarize faint antialiasing noise while preserving stroke edges.
    ink = np.where(ink < 12, 0, ink).astype(np.uint8)
    pil = Image.fromarray(ink, mode="L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    rgb = Image.merge("RGB", (pil, pil, pil))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(rgb).unsqueeze(0).to(DEVICE), pil

if "target" not in st.session_state:
    st.session_state.target = random.choice(CLASSES)
if "score" not in st.session_state:
    st.session_state.score = 0
if "round" not in st.session_state:
    st.session_state.round = 1
if "results" not in st.session_state:
    st.session_state.results = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

judges, missing = load_judges(CLASSES)
if missing:
    st.warning(
        "Model weights are missing for: " + ", ".join(missing) +
        ". Train the models first using `python train.py`. The drawing board will still be shown, "
        "but judging is unavailable until weights exist."
    )

left, right = st.columns([1.35, 1])
with left:
    st.subheader(f"Round {st.session_state.round}")
    st.markdown("### Draw this:")
    st.markdown(f"<h2 style='color:#e85d8f'>{st.session_state.target.title()}</h2>", unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=5,
        stroke_color="#111111",
        background_color="#ffffff",
        height=390,
        width=560,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        return_image_data=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        judge_button = st.button("🧑‍⚖️ Ask the AI judges", type="primary", use_container_width=True)
    with c2:
        clear_button = st.button("↻ Clear / redraw", use_container_width=True)

    if clear_button:
        st.session_state.canvas_key += 1
        st.session_state.results = None
        st.rerun()

    if judge_button:
        if canvas.image_data is None or not np.any(canvas.image_data[:, :, :3] < 245):
            st.info("Draw something on the board before submitting.")
        elif not judges:
            st.error("No trained judge weights were found. Run `python train.py` first.")
        else:
            x, preview = preprocess_canvas(canvas.image_data)
            outputs = {}
            for name, model in judges.items():
                with torch.inference_mode():
                    probs = F.softmax(model(x), dim=1)[0]
                confidence, index = torch.max(probs, dim=0)
                outputs[name] = {
                    "prediction": CLASSES[int(index)],
                    "confidence": float(confidence) * 100,
                }
            st.session_state.results = outputs
            correct = sum(v["prediction"] == st.session_state.target for v in outputs.values())
            st.session_state.score += correct * 10
            st.session_state.last_points = correct * 10
            st.session_state.last_correct = correct

with right:
    st.subheader("Your scoreboard")
    m1, m2 = st.columns(2)
    m1.metric("Total points", st.session_state.score)
    m2.metric("Round", st.session_state.round)
    st.caption("10 points for each judge that predicts the target as its top class.")
    st.divider()
    st.subheader("The judges")
    if st.session_state.results:
        for name in MODEL_SPECS:
            if name not in st.session_state.results:
                st.info(f"{name}: weights not available")
                continue
            result = st.session_state.results[name]
            correct = result["prediction"] == st.session_state.target
            st.markdown(f"**{name}** {'✅' if correct else '❌'}")
            st.write(f"Prediction: **{result['prediction'].title()}**")
            st.progress(min(result["confidence"] / 100, 1.0), text=f"Confidence: {result['confidence']:.1f}%")
        st.success(f"Round points: {getattr(st.session_state, 'last_points', 0)}")
    else:
        st.info("Your judges' predictions will appear here after you submit a drawing.")

    if st.button("➡️ Next round", use_container_width=True):
        st.session_state.round += 1
        st.session_state.target = random.choice(CLASSES)
        st.session_state.results = None
        st.session_state.canvas_key += 1
        st.rerun()

st.divider()
with st.expander("About this prototype"):
    st.write(
        "The game uses a selected subset of Quick, Draw! categories. ResNet-18, VGG-11, and "
        "MobileNetV3-Small are trained locally and loaded once per app session. The prototype "
        "awards 10 points per model whose top prediction matches the prompt; confidence is shown "
        "for transparency but does not currently change the score."
    )
    st.caption("Quick, Draw! Dataset is provided under CC BY 4.0. Credit Google Creative Lab and the dataset contributors.")
