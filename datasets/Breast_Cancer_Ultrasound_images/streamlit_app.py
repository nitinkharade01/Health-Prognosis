import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Class mapping
class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}

# Function to load the combined model
@st.cache_resource
def load_model():
    # URLs for model parts on GitHub
    base_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/splitted_model/"
    model_parts = [f"{base_url}model.h5.part{i:02d}" for i in range(1, 35)]

    # Download and combine model parts
    model_bytes = b''
    for part_url in model_parts:
        response = requests.get(part_url)
        model_bytes += response.content

    # Save the combined model as a temporary file
    temp_model_path = "temp_model.h5"
    with open(temp_model_path, "wb") as f:
        f.write(model_bytes)

    # Load the model
    try:
        # Try loading with TensorFlow >= 2.10 (older versions may have issues)
        model = tf.keras.models.load_model(temp_model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

    return model

# Function to preprocess and make predictions
def predict(image, model):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Adjust the size as per your model requirements
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class, predictions

# Streamlit app
st.set_page_config(page_title="Breast Cancer Detection", layout="wide", page_icon="🔬")

# Header
st.title("🔬 Breast Cancer Detection using Ultrasound Images")

# Sidebar for app info
with st.sidebar:
    st.header("About the App")
    st.write(
        """
        This app helps in classifying breast cancer images as either **Benign**, **Malignant**, or **Normal** based on 
        breast ultrasound images. 
        - **Benign**: Non-cancerous lumps
        - **Malignant**: Cancerous lumps
        - **Normal**: No abnormalities detected
        """
    )
    st.write("Developed by: [Your Name]")
    st.write("Machine Learning Model: DenseNet121")

# Main content
st.markdown("### Upload Your Image Below for Classification")

# File uploader widget
uploaded_file = st.file_uploader("Choose a breast ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Show progress bar while loading the model
    with st.spinner("Loading the model... This might take a few seconds."):
        model = load_model()

    # Show progress bar while making predictions
    with st.spinner("Making prediction..."):
        predicted_class, predictions = predict(image, model)

    # Display prediction result
    st.markdown("### Prediction Results")
    st.subheader(f"**Class: {predicted_class}**")

    # Confidence level visualization
    st.write(f"Prediction confidence: {np.max(predictions[0])*100:.2f}%")
    
    # Display additional insights
    st.markdown("### Additional Insights")
    st.write(f"- **Benign Probability**: {predictions[0][0]*100:.2f}%")
    st.write(f"- **Malignant Probability**: {predictions[0][1]*100:.2f}%")
    st.write(f"- **Normal Probability**: {predictions[0][2]*100:.2f}%")

    # Display next steps
    st.markdown("### Next Steps")
    st.write("""
        If the result is **Malignant**, we strongly recommend consulting a healthcare professional immediately.
        For **Benign** or **Normal** results, regular checkups are advised.
    """)

# Footer Section
st.markdown(""" 
    --- 
    ### About the Technology
    This app uses a **Deep Learning model** built with **DenseNet121** architecture for image classification.
    The model was trained on a large dataset of breast ultrasound images to predict if a given image is benign, malignant, or normal.

    ##### Disclaimer:
    The app serves as a **support tool** for initial detection and should not be relied upon as a diagnostic tool. Always consult a healthcare professional for further assessment.
""")
