from __future__ import annotations

from io import BytesIO

from PIL import Image
import streamlit as st

from config import APP_NAME, APP_VERSION
from inference import predict, prettify_class_name


st.set_page_config(
    page_title=APP_NAME,
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .prediction-card {
            padding: 1.25rem 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(49, 51, 63, 0.15);
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }
        .category-label {
            font-weight: 800;
            font-size: 1.35rem;
            margin-bottom: 0.25rem;
        }
        .source-type-label {
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Water Source Classifier")
st.caption(f"Version {APP_VERSION}")

st.write(
    "Upload a photo of a water source in Ethiopia. "
    "The tool estimates the most likely water-source type and then maps the top prediction "
    "to **improved** or **non-improved**."
)

with st.expander("What do these categories mean?", expanded=True):
    st.markdown(
        """
        - **Improved** means the source is generally more likely to provide safer drinking water.
        - **Non-improved** means the source may have a higher risk of unsafe drinking water.

        """
    )

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Choose a clear photo of the water source.",
)

if uploaded_file is None:
    st.info("Upload an image to see the prediction.")
    st.stop()

try:
    image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")
except Exception:
    st.error("The selected file could not be read as an image. Please try a JPG, PNG, or WEBP file.")
    st.stop()

st.image(image, caption="Uploaded image", use_container_width=True)

with st.spinner("Analyzing image..."):
    try:
        result = predict(image)
    except Exception as exc:
        st.error("The image could not be analyzed.")
        st.exception(exc)
        st.stop()

binary_prediction = result.binary_prediction
binary_prediction_title = {
    "improved": "Improved water source",
    "non-improved": "Non-improved water source",
}.get(binary_prediction, "Unknown improved / non-improved category")

top_source_type_readable = prettify_class_name(result.top_source_type)

st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
st.subheader("Final prediction")

if binary_prediction == "improved":
    st.success(binary_prediction_title)
elif binary_prediction == "non-improved":
    st.error(binary_prediction_title)
else:
    st.warning(binary_prediction_title)

st.write(
    f"The model's top source-type prediction is "
    f"**{top_source_type_readable}** with **{result.confidence:.1%}** confidence."
)

st.markdown(
    f"<p class='small-muted'>{result.explanation}</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Top 3 source-type predictions")

for rank, prediction in enumerate(result.top_predictions, start=1):
    source_type = prettify_class_name(prediction.source_type)
    confidence = prediction.confidence
    binary_class = prediction.binary_class

    if binary_class == "unknown":
        binary_label = "unknown binary category"
    else:
        binary_label = binary_class

    st.markdown(
        f"**{rank}. {source_type}** — {confidence:.1%} confidence → **{binary_label}**"
    )
    st.progress(confidence)

st.caption(
    "The improved / non-improved result is derived from the single most likely source type."
)

st.warning(
    "Important: This result is an estimate based on the image. "
    "It should not replace professional water-quality testing."
)
