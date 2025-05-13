import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random

st.set_page_config(page_title="Defexio Prototype", layout="centered")
st.title("🧪 Defexio Fault Detection Prototype")

uploaded_file = st.file_uploader("Upload a surface image", type=["jpg", "jpeg", "png"])

threshold = st.slider("Detection Threshold", 0, 255, 150)
samples = st.slider("Monte Carlo Samples", 100, 5000, 1000)

def monte_carlo_fault_area(mask, samples):
    height, width = mask.shape
    hits = 0
    for _ in range(samples):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        if mask[y, x] == 255:
            hits += 1
    return (hits / samples) * (width * height)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    _, binary_mask = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    st.image(binary_mask, caption="Detected Fault Area", use_column_width=True)

    area = monte_carlo_fault_area(binary_mask, samples)
    st.success(f"Estimated Fault Area: {int(area)} pixels²")
