"""Streamlit interface for the fabric defect classifier.

Run locally:
    streamlit run app.py

The model and its settings are read from models/. See
notebooks/04_resnet18_transfer_learning.ipynb for how the weights were produced.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from predict import InvalidImage, load_and_check, load_model  # noqa: E402

st.set_page_config(page_title="Fabric Defect Classifier", page_icon="🧵",
                   layout="centered")


@st.cache_resource
def get_model():
    return load_model()


st.title("🧵 Fabric Defect Classifier")
st.caption(
    "ResNet18 fine-tuned on 2,737 fabric images. "
    "Upload a close-up photograph of fabric to classify it."
)

try:
    model, prepare, config, device, weights = get_model()
except SystemExit as exc:
    st.error(str(exc))
    st.stop()

CLASS_NAMES = config["class_names"]
NORMAL = "defect free"

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Review threshold", 0.0, 1.0, 0.95, 0.01,
        help="Predictions below this confidence are flagged for a human to check. "
             "0.95 was chosen by measuring accuracy at different thresholds.",
    )

    st.header("Model")
    st.write(f"**Architecture:** ResNet18 (fine-tuned)")
    st.write(f"**Weights:** `{weights.name}`")
    st.write(f"**Input:** {config['img_size']}×{config['img_size']} RGB")
    st.write(f"**Running on:** {device}")

    st.header("Measured performance")
    st.write("Cross-validated on 2,737 held-out images:")
    st.write("- Macro F1 **0.908**, accuracy **0.965**")
    st.write("- Defect / no-defect recall **99.8%**")
    st.write("- 2 of 1,074 defects missed")

    st.header("Limitations")
    st.caption(
        "Trained on one public dataset. On fabric that looks unlike it — "
        "different colours, weaves or lighting — the model can be confidently "
        "wrong. It is a second opinion for an inspector, not a replacement."
    )

SAMPLES_DIR = REPO / "assets" / "demo"
samples = sorted(SAMPLES_DIR.glob("*")) if SAMPLES_DIR.exists() else []
samples = [p for p in samples if p.suffix.lower() in
           {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]

tmp_path = None
true_label = None

if samples:
    tab_sample, tab_upload = st.tabs(["Try a sample image", "Upload your own"])
else:
    tab_upload = st.container()
    tab_sample = None

def known_label(path):
    """Return the true class if the filename matches one, otherwise None.

    Naming a file after a class (for example Broken_stitch.jpg) lets the app
    check the answer. Any other name is still classified, just without a
    correct/wrong verdict.
    """
    stem = path.stem.replace("_", " ").strip().lower()
    for name in CLASS_NAMES:
        if stem == name.lower():
            return name
    return None


if tab_sample is not None:
    with tab_sample:
        st.caption(
            "Pick an image and press the button. Where the file is named after "
            "a class, the app also shows whether the model got it right."
        )
        choice = st.selectbox(
            "Sample images",
            options=samples,
            format_func=lambda p: p.stem.replace("_", " "),
        )
        st.image(str(choice), width=280)
        if st.button("Classify this sample", type="primary"):
            tmp_path = choice
            true_label = known_label(choice)

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a fabric image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    )
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix, delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        true_label = None

if tmp_path is None:
    st.info(
        "Pick a sample above, or upload your own image. "
        "A close-up filling the frame works best."
        if samples else
        "Upload an image to begin. A close-up filling the frame works best."
    )
    st.stop()

is_uploaded = tmp_path.parent != SAMPLES_DIR

try:
    image, warnings = load_and_check(tmp_path)
except InvalidImage as exc:
    st.error(f"**Cannot classify this file.** {exc}")
    st.stop()

import torch  # noqa: E402

with torch.no_grad():
    probs = model(prepare(image).unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()

order = probs.argsort(descending=True)
best = int(order[0])
label = CLASS_NAMES[best]
confidence = float(probs[best])
is_defect = label != NORMAL
defect_total = float(sum(probs[i] for i, n in enumerate(CLASS_NAMES) if n != NORMAL))

left, right = st.columns([1, 1])

with left:
    st.image(image, caption=f"{image.size[0]}×{image.size[1]}", use_container_width=True)

with right:
    if is_defect:
        st.error(f"### {label}")
    else:
        st.success(f"### {label}")

    if true_label is not None:
        if label == true_label:
            st.success(f"✅ Correct — the true label is **{true_label}**")
        else:
            st.warning(f"❌ Wrong — the true label is **{true_label}**")

    st.metric("Confidence", f"{confidence:.1%}")
    st.metric("Any defect present", f"{defect_total:.1%}",
              help="The eight defect classes added together. The model is often "
                   "surer that something is wrong than about which defect it is.")

    if confidence < threshold:
        st.warning("Flagged for human review — confidence below threshold")
    if warnings:
        for w in warnings:
            st.warning(w)
    if confidence >= threshold and not warnings:
        st.info("Accepted automatically")

st.subheader("All class probabilities")
st.bar_chart(
    {CLASS_NAMES[int(i)]: float(probs[int(i)]) for i in order},
    horizontal=True, height=320,
)

with st.expander("How to read this"):
    st.markdown(
        """
**Confidence** is how sure the model is about its top answer. A high number is
not proof of correctness — on fabric unlike its training data the model can be
confidently wrong.

**Any defect present** adds the eight defect classes together. This is usually
the more reliable figure: the model detects defects at 99.8% recall, while
naming the exact type is harder. If this number is high but the confidence is
low, treat it as "something is wrong here, but check what".

**Flagged for human review** means the top answer fell below the threshold in
the sidebar, or something about the image itself was unusual — very small,
an extreme shape, or not stored in colour.

The two hardest classes are **Vertical** and **horizontal**, which had only 27
and 34 training images and are mostly confused with each other.
        """
    )

if is_uploaded:
    tmp_path.unlink(missing_ok=True)
