"""
app.py
Run it with:  streamlit run app.py
Your browser opens with the app. Upload a road photo to see the result.
"""
import io

import numpy as np
import streamlit as st
from PIL import Image

import config

st.set_page_config(page_title="Road Damage Analyser", page_icon="🛣️", layout="wide")


@st.cache_resource(show_spinner="Loading the model…")
def load_analyzer():
    from analyzer import RoadDamageAnalyzer  # imported here so the page appears quickly
    return RoadDamageAnalyzer()


def to_png_bytes(image_array):
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- title
st.markdown(
    "<h1 style=\"font-family: 'Times New Roman', Times, serif; text-transform: uppercase; "
    "letter-spacing: 0.04em;\">Road Damage Analyser</h1>",
    unsafe_allow_html=True,
)
st.write("Upload a photo of a road surface to find the damage, name its type and "
         "estimate how severe it is.")

if not config.MODEL_PATH.exists():
    st.error(
        "No trained model found. In the VS Code terminal, run `python prepare_data.py` "
        "and then `python train.py`, then refresh this page."
    )
    st.stop()

analyzer = load_analyzer()
uploaded = st.file_uploader("Road photo", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is None:
    st.info("Choose a JPG or PNG photo of a road to start.")
    st.stop()

try:
    image = np.array(Image.open(uploaded).convert("RGB"))
except Exception:
    st.error("That file couldn't be opened as an image. Try a JPG or PNG photo.")
    st.stop()

from analyzer import draw_result  # noqa: E402

with st.spinner("Inspecting the road…"):
    result = analyzer.analyze(image)
    annotated = draw_result(image, result)

# ---------------------------------------------------------------- results
left, right = st.columns(2)
left.image(image, caption="Your photo", width="stretch")
right.image(annotated, caption="Detected damage", width="stretch")

st.divider()

c1, c2 = st.columns(2)
c1.metric("Damage type", result["label"])
c2.metric("Severity", result["severity"])

st.download_button(
    "Download annotated photo",
    data=to_png_bytes(annotated),
    file_name=f"road_damage_{result['class_name']}.png",
    mime="image/png",
)