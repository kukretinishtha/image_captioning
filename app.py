import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "Nishthaaa/image_captioning"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Streamlit UI
st.title("Cartoon Caption Generator 🖼️📜")
st.write("Upload a cartoon image and get a funny caption!")

uploaded_file = st.file_uploader("Upload a Cartoon Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)


    # Preprocess and generate caption
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("Generated Caption:")
    st.write(f"💬 {caption}")
