import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import models, transforms

APP_DIR = Path(__file__).parent
WEIGHTS_DIR = APP_DIR / "weights"
CLASSES_FILE = APP_DIR / "classes.txt"
IMAGE_SIZE = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NEW: game settings (previously the game had no end and the 10 was hard-coded)
TOTAL_ROUNDS = 5
POINTS_PER_JUDGE = 10

# NEW: pen colours. Only vivid/dark colours on purpose: very light colours (yellow, pastel pink)
# would give the models faint strokes and hurt their predictions.
PEN_COLORS = {
    "⚫ Black": "#111111",
    "🔴 Red": "#E63946",
    "🔵 Blue": "#1D4ED8",
    "🟢 Green": "#16A34A",
    "🟠 Orange": "#F97316",
    "🟣 Purple": "#7E22CE",
}

MODEL_SPECS = {
    "ResNet": ("resnet18", "resnet18.pt"),
    "VGG-16": ("vgg16", "vgg16.pt"),
    "MobileNet": ("mobilenet_v3_small", "mobilenet_v3_small.pt"),
}
ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "vgg16": "VGG-16",
    "mobilenet_v3_small": "MobileNetV3-Small",
}

def join_names(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]

st.set_page_config(page_title="AI Draw Battle", page_icon="🎨", layout="wide")

# ---------------------------------------------------------------------------
# STYLE (presentation only: no game logic lives here)
# ---------------------------------------------------------------------------
JUDGE_COLORS = {"ResNet": "#2563EB", "VGG-16": "#64748B", "MobileNet": "#0F766E"}

def html(markup):
    """Render raw HTML (lines are stripped so Markdown never mistakes indentation for code)."""
    st.markdown("".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)

st.markdown("""
<style>
html, body, [class*="st-"], .stApp {font-family: Arial, sans-serif;}
.stApp {background: #f7f8fa; color: #1f2937;}
.block-container {max-width: 1180px; padding-top: 1.5rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 12px; background: #ffffff;
       border: 1px solid #e5e7eb; margin-bottom: 1.2rem;}
.hero h1 {margin:0; color:#111827; font-weight:800; font-size:2.2rem;}
.hero p {margin:.35rem 0 0; color:#4b5563; font-size:1rem;}
.pills {display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.9rem;}
.pill {background:#f3f4f6; border:1px solid #e5e7eb; border-radius:6px; padding:.25rem .7rem;
       font-weight:700; font-size:.85rem; color:#374151;}
.pill i {display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.4rem;}
.target {background:#fff; border:2px dashed #9ca3af; border-radius:12px; text-align:center; padding:.9rem 1rem;
         margin:.2rem 0 1rem;}
.target .kicker {font-size:.8rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#6b7280;}
.target .word {font-size:3rem; font-weight:800; line-height:1.1; color:#1f2937;}
.jcard {border-radius:10px; padding:.85rem 1rem; margin-bottom:.7rem; border:1px solid #e5e7eb; background:#fff;}
.jcard.ok {background:#f0fdf4; border-color:#86efac;}
.jcard.no {background:#fff; border-color:#d1d5db;}
.jtop {display:flex; justify-content:space-between; align-items:center;}
.jname {font-weight:700; color:#111827; font-size:1rem;}
.jbadge {font-weight:700; font-size:.8rem; border-radius:5px; padding:.15rem .5rem;}
.jcard.ok .jbadge {background:#dcfce7; color:#166534;} .jcard.no .jbadge {background:#f3f4f6; color:#4b5563;}
.jguess {font-size:1.5rem; font-weight:800; color:#111827; line-height:1.2; margin:.15rem 0 .35rem;}
.jbar {height:8px; background:#e5e7eb; border-radius:4px; overflow:hidden;}
.jbar i {display:block; height:100%; border-radius:4px;}
.jconf {font-size:.82rem; font-weight:600; color:#6b7280; margin-top:.25rem;}
.jempty {border:1px dashed #d1d5db; border-radius:10px; padding:1.1rem; text-align:center; color:#6b7280; background:#fff;}
.points {border-radius:10px; padding:.75rem 1rem; text-align:center; font-weight:800; font-size:1.2rem; margin:.4rem 0 .8rem;
         background:#166534; color:#fff;}
.points.zero {background:#6b7280;}
.over {text-align:center; border-radius:12px; padding:1.3rem 1rem; margin-bottom:1rem; color:#111827;
       background:#fff; border:1px solid #e5e7eb;}
.over .trophy {font-size:2.5rem; line-height:1;} .over .t {font-size:1.6rem; font-weight:800;} .over .s {font-weight:600; color:#4b5563;}
[data-testid="stMetric"] {background:#fff; border:1px solid #e5e7eb; padding:12px 16px; border-radius:10px;}
[data-testid="stMetricValue"] {font-weight:800; color:#111827;}
[data-testid="stMetricLabel"] {font-weight:600; color:#6b7280;}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {border-radius:8px; font-weight:700; padding:.5rem 1rem;}
[data-testid="stBaseButton-primary"] {background:#2563eb !important; border:1px solid #2563eb !important; color:#fff !important;}
[data-testid="stBaseButton-secondary"] {border:1px solid #d1d5db;}
[data-testid="stBaseButton-primary"]:hover {background:#1d4ed8 !important;}
[data-testid="stBaseButton-primary"]:disabled {background:#e5e7eb !important; color:#9ca3af !important; border-color:#e5e7eb !important;}
.st-key-toolbar, .st-key-canvas_card {border-radius:10px; background:#fff; border:1px solid #e5e7eb; margin-bottom:.4rem;}
h2, h3 {font-weight:800 !important; color:#111827;}
@media (max-width: 768px) {.hero h1 {font-size:1.7rem;} .target .word {font-size:2.4rem;}}
</style>
""", unsafe_allow_html=True)

_pills = "".join(f'<span class="pill"><i style="background:{JUDGE_COLORS.get(n, "#6c63ff")}"></i>{n}</span>' for n in MODEL_SPECS)
html(f"""
<div class="hero">
<h1>🎨 AI Draw Battle</h1>
<p>Draw the target. Can you convince {join_names(MODEL_SPECS)}?</p>
<div class="pills">{_pills}</div>
</div>
""")

def read_classes():
    if CLASSES_FILE.exists():
        return [x.strip() for x in CLASSES_FILE.read_text().splitlines() if x.strip()]
    return ["apple", "cat", "airplane", "fish", "house", "tree", "star", "bicycle"]

CLASSES = read_classes()
@st.cache_resource
def load_judges(class_names):
    loaded = {}
    missing = []
    errors = {}  # NEW: weights that exist but don't match classes.txt
    for label, (arch, filename) in MODEL_SPECS.items():
        path = WEIGHTS_DIR / filename
        if not path.exists():
            missing.append(label)
            continue
        if arch == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        elif arch == "vgg16":
            model = models.vgg16(weights=None)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(class_names))
        else:
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(class_names))
        try:  # CHANGED: a wrong-sized checkpoint no longer crashes the whole app
            checkpoint = torch.load(path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state"])
        except Exception as exc:
            errors[label] = str(exc).splitlines()[0][:140]
            continue
        model.to(DEVICE).eval()
        loaded[label] = model
    return loaded, missing, errors

def preprocess_canvas(canvas_rgba):
    # Canvas is white with dark ink; QuickDraw bitmap drawings are normalized to dark background/light strokes.
    image = Image.fromarray(canvas_rgba.astype("uint8")).convert("RGB")  # CHANGED: dropped deprecated mode arg
    # CHANGED (for coloured pens): ink strength comes from the DARKEST colour channel instead of luminance.
    # For black / white / gray pixels this is identical to the old grayscale; for red, green, orange... it
    # keeps the stroke at full strength instead of making it faint.
    arr = np.asarray(image).min(axis=2)
    # Remove near-white canvas and invert to dark background with bright strokes.
    ink = 255 - arr
    # Binarize faint antialiasing noise while preserving stroke edges.
    ink = np.where(ink < 12, 0, ink).astype(np.uint8)
    pil = Image.fromarray(ink).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)  # CHANGED: dropped mode arg
    rgb = Image.merge("RGB", (pil, pil, pil))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(rgb).unsqueeze(0).to(DEVICE), pil

# ---------------------------------------------------------------------------
# NEW: game state. Everything about the game lives in session_state, and the
# score is *derived* from the round history so it can never get out of sync.
# ---------------------------------------------------------------------------
def new_game():
    # Words are drawn from a shuffled deck: no repeats within a game.
    st.session_state.deck = random.sample(CLASSES, min(TOTAL_ROUNDS, len(CLASSES)))
    st.session_state.round = 1
    st.session_state.target = st.session_state.deck[0]
    st.session_state.results = None
    st.session_state.judged = False      # has THIS round been scored yet?
    st.session_state.finished = False
    st.session_state.history = []        # one entry per judged round
    st.session_state.canvas_key = st.session_state.get("canvas_key", 0) + 1

if "history" not in st.session_state:
    new_game()

def total_rounds():
    return len(st.session_state.deck)

def total_score():
    return sum(h["points"] for h in st.session_state.history)

def history_rows():
    """Rows for the scoreboard table: one row per judged round."""
    rows = []
    for h in st.session_state.history:
        row = {"Round": h["round"], "Word": h["target"].title()}
        for name in MODEL_SPECS:
            out = h["outputs"].get(name)
            if out is None:
                row[name] = "—"
            elif out["prediction"] == h["target"]:
                row[name] = "✅"
            else:
                row[name] = f"❌ {out['prediction']}"
        row["Points"] = h["points"]
        rows.append(row)
    return rows

def render_footer():
    st.divider()
    with st.expander("About this prototype"):
        st.write(
            "The game uses 8 selected Quick, Draw! categories, configured for 3,000 training drawings per class. "
            + join_names(ARCH_LABELS[a] for a, _ in MODEL_SPECS.values())
            + " are trained locally and loaded once per app session. "
            f"The prototype awards {POINTS_PER_JUDGE} points per model whose top prediction matches the "
            "prompt; confidence is shown for transparency but does not currently change the score."
        )
        st.caption("Quick, Draw! Dataset is provided under CC BY 4.0. Credit Google Creative Lab and the dataset contributors.")

judges, missing, errors = load_judges(tuple(CLASSES))
if missing:
    st.warning(
        "Model weights are missing for: " + ", ".join(missing) +
        ". Train the models first using `python train.py`. The drawing board will still be shown, "
        "but judging is unavailable until weights exist."
    )
if errors:  # NEW
    st.error(
        "These weights don't match classes.txt (were they trained on a different category list?): "
        + "; ".join(f"{k}: {v}" for k, v in errors.items())
        + ". Retrain them with the current classes.txt."
    )

# NEW: game-over screen replaces the board once the last round is finished.
if st.session_state.finished:
    html(f"""
    <div class="over"><div class="trophy">🏆</div><div class="t">Game over!</div>
    <div class="s">You finished all {total_rounds()} rounds. Here is how it went.</div></div>
    """)
    n_correct = sum(o["prediction"] == h["target"] for h in st.session_state.history for o in h["outputs"].values())
    n_total = sum(len(h["outputs"]) for h in st.session_state.history)
    g1, g2, g3 = st.columns(3)
    g1.metric("Final score", total_score())
    g2.metric("Rounds played", len(st.session_state.history))
    g3.metric("Judge hit rate", f"{(100 * n_correct / n_total) if n_total else 0:.0f}%")
    st.dataframe(history_rows(), hide_index=True, width="stretch")
    if st.button("🔄 Play again", type="primary", width="stretch"):
        new_game()
        st.rerun()
    render_footer()
    st.stop()

left, right = st.columns([1.35, 1])
with left:
    st.subheader(f"Round {st.session_state.round} of {total_rounds()}")  # CHANGED: shows "of N"
    done = len(st.session_state.history)
    st.progress(done / total_rounds(), text=f"{done} of {total_rounds()} rounds finished")
    html(f"""
    <div class="target"><div class="kicker">🎯 Draw this</div>
    <div class="word">{st.session_state.target.title()}</div></div>
    """)
    # NEW: drawing tools (pen / eraser, colour, brush size)
    with st.container(border=True, key="toolbar"):
        tool_col, size_col = st.columns([1, 1.3])
        with tool_col:
            tool = st.radio("Tool", ["✏️ Pen", "🧽 Eraser"], horizontal=True, key="tool",
                            disabled=st.session_state.judged)
        with size_col:
            brush = st.slider("Brush size", 2, 30, 5, key="brush_size", disabled=st.session_state.judged)
        is_eraser = tool.endswith("Eraser")
        color_name = st.selectbox(
            "Pen colour",
            list(PEN_COLORS),
            key="pen_color",
            disabled=st.session_state.judged or is_eraser,
            help="Choose a colour from the dropdown."
        )
    # The eraser paints white (the canvas background) and is 2x wider than the pen.
    stroke_color = "#FFFFFF" if is_eraser else PEN_COLORS[color_name]
    stroke_width = brush * 2 if is_eraser else brush
    with st.container(border=True, key="canvas_card"):
        canvas = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#ffffff",
            height=390,
            width=560,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            return_image_data=True,
        )
    c1, c2 = st.columns(2)
    with c1:
        # CHANGED: locked once the round is judged (stops re-judging to farm points)
        judge_button = st.button("🧑‍⚖️ Ask the AI judges", type="primary", width="stretch",
                                 disabled=st.session_state.judged)
    with c2:
        clear_button = st.button("↻ Clear / redraw", width="stretch", disabled=st.session_state.judged)
    if st.session_state.judged:
        st.caption("🔒 This round is judged. Press **Next round** to continue.")

    if clear_button and not st.session_state.judged:
        st.session_state.canvas_key += 1
        st.session_state.results = None
        st.rerun()

    if judge_button and not st.session_state.judged:
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
            correct = sum(v["prediction"] == st.session_state.target for v in outputs.values())
            # CHANGED: points are recorded exactly once per round, in the history
            st.session_state.results = outputs
            st.session_state.judged = True
            st.session_state.history.append({
                "round": st.session_state.round,
                "target": st.session_state.target,
                "outputs": outputs,
                "points": correct * POINTS_PER_JUDGE,
            })
            st.rerun()  # refresh so the scoreboard and locked buttons update immediately

with right:
    st.subheader("Your scoreboard")
    m1, m2 = st.columns(2)
    m1.metric("Total points", total_score())
    m2.metric("Round", f"{st.session_state.round} / {total_rounds()}")  # CHANGED: "x / N"
    st.caption(f"{POINTS_PER_JUDGE} points for each judge that predicts the target as its top class "
               f"(max {POINTS_PER_JUDGE * len(MODEL_SPECS)} per round).")
    if st.session_state.history:  # NEW: per-round history table
        st.dataframe(history_rows(), hide_index=True, width="stretch")
    st.divider()
    st.subheader("The judges")
    if st.session_state.results:
        n_cols = 2 if len(MODEL_SPECS) > 3 else 1  # CHANGED: 2x2 grid only when a 4th judge is added
        judge_cols = st.columns(n_cols)
        for i, name in enumerate(MODEL_SPECS):
            with judge_cols[i % n_cols]:
                if name not in st.session_state.results:
                    html(f'<div class="jempty">{name}: weights not available</div>')
                    continue
                result = st.session_state.results[name]
                correct = result["prediction"] == st.session_state.target
                color = JUDGE_COLORS.get(name, "#6c63ff")
                html(f"""
                <div class="jcard {'ok' if correct else 'no'}">
                <div class="jtop"><span class="jname">{name}</span>
                <span class="jbadge">{'✅ Correct' if correct else '❌ Missed'}</span></div>
                <div class="jguess">{result['prediction'].title()}</div>
                <div class="jbar"><i style="width:{min(result['confidence'], 100):.1f}%;background:{color}"></i></div>
                <div class="jconf">Confidence: {result['confidence']:.1f}%</div>
                </div>
                """)
        pts = st.session_state.history[-1]["points"]
        if pts:
            html(f'<div class="points">+{pts} points this round 🎉</div>')
        else:
            html('<div class="points zero">No points this round</div>')
    else:
        html('<div class="jempty">🧑‍⚖️ Your judges\' predictions will appear here after you submit a drawing.</div>')

    # CHANGED: can't skip a round without judging; last round becomes "Finish game"
    is_last = st.session_state.round >= total_rounds()
    next_label = "🏁 Finish game" if is_last else "➡️ Next round"
    if st.button(next_label, width="stretch", disabled=not st.session_state.judged):
        if is_last:
            st.session_state.finished = True
        else:
            st.session_state.round += 1
            st.session_state.target = st.session_state.deck[st.session_state.round - 1]
            st.session_state.results = None
            st.session_state.judged = False
            st.session_state.canvas_key += 1
        st.rerun()

render_footer()
