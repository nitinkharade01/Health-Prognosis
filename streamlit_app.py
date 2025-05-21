import os
import json
import pickle
import requests
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------- Page Configuration -------------------------
st.set_page_config(page_title="Health Prognosis", layout="wide")

# Load configuration
with open("config.json", encoding="utf-8") as config_file:
    config_params = json.load(config_file)['params']

# ------------------------- UI Title -------------------------
st.title(config_params["app_name"])

# ------------------------- Constants -------------------------
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
MODEL_PATH = 'Diasease_model.h5'

# ------------------------- Helper Functions -------------------------
@st.cache_resource
def load_breast_cancer_model():
    base_url = "_"
    model_parts = [f"{base_url}model.h5.part{i:02d}" for i in range(1, 35)]
    model_bytes = b''

    for part_url in model_parts:
        response = requests.get(part_url)
        model_bytes += response.content

    temp_model_path = "temp_model.h5"
    with open(temp_model_path, "wb") as f:
        f.write(model_bytes)

    try:
        model = tf.keras.models.load_model(temp_model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e

    return model

def predict_breast_cancer(image, model):
    img_array = tf.image.resize(np.array(image), (256, 256))
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = CLASS_MAPPING[np.argmax(predictions[0])]
    return predicted_class, predictions

def predict_skin_disease(file_path):
    model = load_model(MODEL_PATH, compile=False)
    optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    labels = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    prediction = np.argmax(model.predict(img_array), axis=1)[0]
    
    return labels[prediction]

def predict_diabetes(features):
    with open('RFmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict([features])[0]
    return (
        "🩺 You are a diabetic person. Regular monitoring and lifestyle strategies are essential."
        if prediction == 1 else
        "✅ Relax! You are not a diabetic person."
    )

def predict_heart_disease(features):
    with open('RFmodel_heart.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict([features])[0]
    return (
        "💔 You have a heart problem. Seek medical advice immediately."
        if prediction == 1 else
        "❤️ Relax! You have a healthy heart."
    )

# Load model and tokenizer once
@st.cache_resource
def load_chatbot_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_chatbot_model()

# Function to get chatbot response
def get_chatbot_response(user_input):
    inputs = tokenizer(
        user_input,
        return_tensors='pt',
        truncation=True,
        max_length=1024
    ).input_ids.to(device)

    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------- Sidebar Navigation -------------------------
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choose a Model",
    ["Home", "About", "Breast Cancer Detection", "Skin Disease Prediction",
     "Diabetes Prediction", "Heart Disease Prediction"]
)

# ------------------------- Pages -------------------------
# Home Page
if page == "Home":
    st.header("Welcome to the Health Prediction App")
    st.markdown(f"""
        <div style="font-size:18px;">{config_params['description']}</div>
        <div style="font-size:16px; color: gray;">
            Get personalized predictions for skin diseases, diabetes, and heart health using advanced machine learning models.
        </div>
    """, unsafe_allow_html=True)

    # Chatbot UI
    st.header("🤖 Sam")
    st.markdown("Ask me health-related questions!")
    
    # Get user input and display response
    user_input = st.text_input("You:", "")
    if user_input:
        response = get_chatbot_response(user_input)
        st.text_area("Chatbot:", value=response, height=150)

# About Page
elif page == "About":
    st.header("About the App")
    st.markdown(f"""
        <ul style="font-size: 18px;">
            <li>🧴 <strong>Skin Disease Prediction</strong>: Upload an image to detect Acne, Melanoma, and more.</li>
            <li>🩺 <strong>Diabetes Prediction</strong>: Predict based on your health features.</li>
            <li>❤️ <strong>Heart Disease Prediction</strong>: Estimate risk based on health parameters.</li>
            <li>🤖 <strong>Chatbot</strong>: Ask health-related questions anytime.</li>
            <li>👨‍💻 <strong>Developer</strong>: {config_params['developer']}</li>
        </ul>
    """, unsafe_allow_html=True)

# Breast Cancer Detection Page
elif page == "Breast Cancer Detection":
    st.header("Breast Cancer Detection 🔬")
    st.markdown("Upload an ultrasound image to predict whether the tumor is **Benign**, **Malignant**, or **Normal**.")
    
    uploaded_file = st.file_uploader("Choose an ultrasound image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_file = Image.open(uploaded_file)
        st.image(image_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading model..."):
            model = load_breast_cancer_model()

        with st.spinner("Making prediction..."):
            predicted_class, predictions = predict_breast_cancer(image_file, model)

        st.subheader(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence level: {np.max(predictions[0])*100:.2f}%")
        st.write(f"Benign Probability: {predictions[0][0]*100:.2f}%")
        st.write(f"Malignant Probability: {predictions[0][1]*100:.2f}%")
        st.write(f"Normal Probability: {predictions[0][2]*100:.2f}%")

# Skin Disease Prediction Page
elif page == "Skin Disease Prediction":
    st.header("Skin Disease Prediction 🧴")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        prediction = predict_skin_disease(file_path)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: {prediction}")

# Diabetes Prediction Page
elif page == "Diabetes Prediction":
    st.header("Diabetes Prediction 🩺")
    st.markdown("Fill out the details below to predict your risk of diabetes.")

    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    if st.button("Predict Diabetes"):
        result = predict_diabetes(features)
        st.success(result)

# Heart Disease Prediction Page
elif page == "Heart Disease Prediction":
    st.header("Heart Disease Prediction ❤️")
    st.markdown("Fill out the details below to check your risk of heart disease.")

    # Input fields
    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0)
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

    sex_val = 1 if sex == "Male" else 0
    features = [age, sex_val, cp, trestbps, chol, fbs, thalach, exang, oldpeak, ca, thal]

    if st.button("Predict Heart Disease"):
        result = predict_heart_disease(features)
        st.success(result)
