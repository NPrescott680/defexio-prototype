import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Defexio Fault Estimator", layout="centered")
st.title("🧪 Defexio: Surface Fault Estimator")

uploaded_file = st.file_uploader("Upload a surface image", type=["jpg", "jpeg", "png"])

# User inputs
threshold = st.slider("Detection Threshold", 0, 255, 150)
samples = st.slider("Monte Carlo Samples", 100, 5000, 1000)
trials = st.slider("Number of Trials for Statistics", 10, 100, 30)

# Function to calculate area with statistics
def monte_carlo_area_with_stats(mask, samples=1000, trials=30):
    height, width = mask.shape
    total_pixels = width * height
    results = []

    for _ in range(trials):
        hits = 0
        for _ in range(samples):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if mask[y, x] == 255:
                hits += 1
        estimated_area = (hits / samples) * total_pixels
        results.append(estimated_area)

    results = np.array(results)
    mean_area = np.mean(results)
    std_dev = np.std(results)
    variance = np.var(results)
    conf_int = stats.norm.interval(0.95, loc=mean_area, scale=std_dev / np.sqrt(trials))

    return mean_area, std_dev, variance, conf_int, results

# Run app logic if image uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)
    st.image(image, caption="Original Image", use_container_width=True)

    _, binary_mask = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    st.image(binary_mask, caption="Detected Fault Area", use_container_width=True)

    mean_area, std_dev, variance, conf_int, all_estimates = monte_carlo_area_with_stats(
        binary_mask, samples=samples, trials=trials
    )

    # Display results
    st.success(f"📐 Estimated Fault Area: {int(mean_area)} pixels²")
    st.info(f"📊 Std Dev: ±{int(std_dev)} pixels²")
    st.caption(f"Variance: {int(variance)} pixels²")
    st.markdown(f"📎 95% Confidence Interval: **{int(conf_int[0])}–{int(conf_int[1])}** pixels²")

    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(all_estimates, bins=20, color="#69b3a2", edgecolor="black")
    ax.axvline(mean_area, color="red", linestyle="dashed", linewidth=1, label="Mean")
    ax.set_title("Distribution of Estimated Fault Areas")
    ax.set_xlabel("Area (pixels²)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

