"""
app.py
Step 3 of the project. Run it with:  streamlit run app.py
Your browser opens with the app. Upload a road photo to see the result.
"""
import io

import numpy as np
import streamlit as st
from PIL import Image

import config

st.set_page_config(page_title="Road Damage Inspector", page_icon="🛣️", layout="wide")

SEVERITY_STYLE = {
    "Low": ("#2E8B57", "Minor surface damage. Monitor it at the next routine inspection."),
    "Medium": ("#C27A00", "Noticeable damage. Schedule a repair before it spreads."),
    "High": ("#C8281E", "Serious damage. Prioritise this for repair."),
}


@st.cache_resource(show_spinner="Loading the model…")
def load_analyzer():
    from analyzer import RoadDamageAnalyzer  # imported here so the page appears quickly
    return RoadDamageAnalyzer()


def to_png_bytes(image_array):
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("How it works")
    st.write(
        "A MobileNetV2 model identifies the type of damage. Grad-CAM shows where the model "
        "looked, and that area becomes the box. Severity comes from how much of the photo "
        "the damaged area covers."
    )
    st.markdown(
        f"- **Low**: under {config.SEVERITY_LOW_BELOW:.0%} of the photo\n"
        f"- **Medium**: {config.SEVERITY_LOW_BELOW:.0%} to {config.SEVERITY_MEDIUM_BELOW:.0%}\n"
        f"- **High**: over {config.SEVERITY_MEDIUM_BELOW:.0%}"
    )
    show_heatmap = st.toggle("Show where the model looked", value=False)
    st.caption(
        "Boxes are estimates from Grad-CAM, not a trained detector, and the app marks one "
        "defect per photo."
    )

# ---------------------------------------------------------------- main page
st.title("Road Damage Inspector")
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

from analyzer import draw_result, overlay_heatmap  # noqa: E402

with st.spinner("Inspecting the road…"):
    result = analyzer.analyze(image)
    annotated = draw_result(image, result)

left, right = st.columns(2)
left.image(image, caption="Your photo", width="stretch")
right.image(annotated, caption="Detected damage", width="stretch")

if show_heatmap:
    st.image(overlay_heatmap(image, result["heatmap"]),
             caption="Grad-CAM heatmap: red areas influenced the prediction most",
             width="stretch")

st.divider()

colour, advice = SEVERITY_STYLE[result["severity"]]
c1, c2, c3 = st.columns(3)
c1.metric("Damage type", result["label"])
c2.metric("Confidence", f"{result['confidence']:.0%}")
c3.metric("Severity", result["severity"])

st.markdown(
    f"<div style='border-left: 6px solid {colour}; padding: 0.6rem 1rem; "
    f"background: rgba(0,0,0,0.03); border-radius: 4px;'>"
    f"<strong style='color:{colour}'>{result['severity']} severity.</strong> {advice} "
    f"The damaged area covers about {result['area_ratio']:.0%} of the photo.</div>",
    unsafe_allow_html=True,
)

if result["confidence"] < config.LOW_CONFIDENCE:
    st.warning(
        "The model isn't sure about this one. The photo may be unclear, show more than one "
        "type of damage, or show something the model wasn't trained on."
    )

st.subheader("Scores for each damage type")
for name, prob in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
    st.progress(min(prob, 1.0), text=f"{name}: {prob:.0%}")
    
st.download_button(
    "Download annotated photo",
    data=to_png_bytes(annotated),
    file_name=f"road_damage_{result['class_name']}.png",
    mime="image/png",
)
