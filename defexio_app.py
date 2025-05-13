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

def monte_carlo_area_with_stats(mask, samples=1000, trials=30):
    height, width = mask.shape
    pixel_area = width * height
    results = []

    for _ in range(trials):
        hits = 0
        for _ in range(samples):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if mask[y, x] == 255:
                hits += 1
        estimated_area = (hits / samples) * pixel_area
        results.append(estimated_area)

    mean_area = np.mean(results)
    std_dev = np.std(results)
    variance = np.var(results)

    return mean_area, std_dev, variance


if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)
    st.image(image, caption="Original Image", use_container_width=True)

    _, binary_mask = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    st.image(binary_mask, caption="Detected Fault Area", use_container_width=True)

    mean_area, std_dev, variance = monte_carlo_area_with_stats(binary_mask, samples=samples, trials=30)

    st.success(f"Estimated Fault Area: {int(mean_area)} pixels²")
    st.info(f"Standard Deviation: ±{int(std_dev)} pixels²")
    st.caption(f"Variance: {int(variance)} pixels²")
