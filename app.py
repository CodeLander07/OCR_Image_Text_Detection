import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import matplotlib.pyplot as plt
import numpy as np
import tempfile


# Custom CSS Styling

st.markdown("""
    <style>

    /* Page background */
    .main {
        background-color: #f5f7fa;
    }

    /* Title styling */
    .title {
        font-size: 40px;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 18px;
        color: #34495E;
        text-align: center;
    }

    /* Card container */
    .card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }

    /* Upload box */
    .upload-box {
        background: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        text-align: center;
        border: 2px dashed #3498db;
    }

    /* Extracted text box */
    .text-box {
        background: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        color: #2c3e50;
        white-space: pre-wrap;
    }

    </style>
""", unsafe_allow_html=True)


# Load OCR model once

@st.cache_resource
def load_model():
    return ocr_predictor(pretrained=True)

model = load_model()


# Sidebar

st.sidebar.title("⚙️")
assume_straight = st.sidebar.checkbox("Assume Straight Pages", value=True)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)


# Title Section

st.markdown("<h1 class='title'>OCR++: Intelligent Scene Text Recognition </h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Extract text from images & PDFs instantly using docTR</p>", unsafe_allow_html=True)

st.write("")


# File Upload UI

with st.container():


    uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    st.markdown("</div>", unsafe_allow_html=True)


# OCR Processing

if uploaded_file is not None:

    with st.spinner("Processing your document..."):

        # Store file temp
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(uploaded_file.read())
            file_path = temp.name

        # Load docTR document
        if uploaded_file.type == "application/pdf":
            doc = DocumentFile.from_pdf(file_path)
        else:
            doc = DocumentFile.from_images(file_path)
            st.image(file_path, caption="Uploaded Image", use_container_width=True)

        # Apply OCR
        result = model(doc)

  
    # Extracted Text UI
  
    st.subheader("Extracted Text")

    extracted_text = ""

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    extracted_text += word.value + " "

    st.markdown(f"<div class='text-box'>{extracted_text}</div>", unsafe_allow_html=True)
  

  
    # Bounding Boxes UI
  
    if show_boxes:
    
        st.subheader("Text Detection Boxes")
       

        plt.figure(figsize=(10, 10))
        result.show()
        st.pyplot()
       
