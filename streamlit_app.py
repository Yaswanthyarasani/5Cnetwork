import streamlit as st
import requests
from PIL import Image
import io

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload an MRI image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image.", use_column_width=True)

    if st.button("Segment Image"):
        # Send image to FastAPI backend
        files = {'file': uploaded_file}
        response = requests.post("http://localhost:8000/predict/", files=files)
        
        # Display segmentation result
        segmented_image = Image.open(io.BytesIO(response.content))
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)
