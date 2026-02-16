import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Adham | Aerial Detection", page_icon="🚁", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTitle { color: #1E3A8A; font-family: 'Helvetica'; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚁 Aerial Object Detection System")
st.write("Graduation Project by: **Adham**")
st.sidebar.header("Model Settings")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Original Image")
            st.image(img, use_container_width=True)
            
        with col2:
            results = model.predict(img, conf=conf_threshold)
            res_plotted = results[0].plot()
            st.success("Detection Result")
            st.image(res_plotted, use_container_width=True)
            
        st.divider()
        boxes = results[0].boxes
        if len(boxes) > 0:
            st.subheader(f"🔍 Detection Details (Found {len(boxes)} objects)")
            for box in boxes:
                label = model.names[int(box.cls)]
                prob = float(box.conf)
                st.write(f"📍 Found **{label}** with confidence: **{prob:.2%}**")
        else:
            st.warning("No Drones or Birds detected.")

except Exception as e:
    st.error(f"Error: {e}")