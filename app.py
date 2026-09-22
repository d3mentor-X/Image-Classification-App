import os
import streamlit as st
from PIL import Image
from google import genai

st.set_page_config(
    page_title="AI Image Identifier",
    layout="centered"
)

st.title("AI Image Identifier")
st.markdown("Academic image identification powered by the Google Gemini API.")

# Read the API key from the environment variable (do not hardcode or display)
api_key = os.environ.get("GEMINI_API_KEY")


def analyze_image_with_gemini(image: Image.Image, api_key: str) -> str:
    """Sends the provided image to Gemini for detailed academic analysis."""
    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze this image in detail. Identify the main subject or objects, "
        "describe their appearance, explain the scene, and provide a useful, "
        "detailed academic description including important visible details."
    )
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[image, prompt]
    )
    return response.text


# ----------------------------------------------------
# 1. Image Upload Section
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        if not api_key:
            st.error("The `GEMINI_API_KEY` environment variable is not set. Please set it before analyzing images.")
        else:
            with st.spinner("Analyzing uploaded image with Gemini... Please wait."):
                try:
                    result_text = analyze_image_with_gemini(image, api_key)
                    st.subheader("Analysis Result")
                    st.markdown(result_text)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

# ----------------------------------------------------
# 2. Camera Input Section
# ----------------------------------------------------
st.write("---")
st.subheader("Capture Photo with Camera")

camera_file = st.camera_input("Take a photo")

if camera_file is not None:
    camera_image = Image.open(camera_file).convert("RGB")
    st.image(camera_image, caption="Captured Photo", use_container_width=True)

    if st.button("Analyze Camera Image"):
        if not api_key:
            st.error("The `GEMINI_API_KEY` environment variable is not set. Please set it before analyzing images.")
        else:
            with st.spinner("Analyzing captured photo with Gemini... Please wait."):
                try:
                    result_text = analyze_image_with_gemini(camera_image, api_key)
                    st.subheader("Analysis Result")
                    st.markdown(result_text)
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
