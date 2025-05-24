import os
import json
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import random
import io
import base64
import gdown
import requests

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="Health Prognosis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom theme configuration
st.markdown("""
    <style>
        /* Main background */
        .main {
            background-color: #121212;
            background-image: linear-gradient(45deg, #121212 0%, #1e1e1e 100%);
        }
        .stApp {
            background-color: #121212;
            background-image: linear-gradient(45deg, #121212 0%, #1e1e1e 100%);
        }
        
        /* Text colors */
        h1, h2, h3 {
            color: #eeeeee !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        .stMarkdown {
            color: #eeeeee !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        p, div {
            color: #eeeeee !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Cards and Containers */
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #1a1a1a !important;
            border: 1px solid #333 !important;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
            border-radius: 5px;
        }
        .stTextInput>div>div>input:focus {
            border-color: #666 !important;
            box-shadow: 0 0 0 0.25rem rgba(255, 255, 255, 0.1);
        }
        
        /* Number inputs */
        .stNumberInput>div>div>input {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
            border-radius: 5px;
        }
        
        /* Select boxes */
        .stSelectbox>div>div>select {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
            border-radius: 5px;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #111111 !important;
            border-color: #111111 !important;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        
        /* --- Robust File Uploader Styling --- */
        section[data-testid="stFileUploader"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        section[data-testid="stFileUploader"] div[data-testid="stFileDropzone"] {
            background: #181818 !important;
            border: 2px dashed #222 !important;
            border-radius: 10px !important;
            color: #eeeeee !important;
        }
        section[data-testid="stFileUploader"] button {
            background: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            padding: 6px 18px !important;
            margin: 0 !important;
            font-size: 1rem !important;
            min-width: 0 !important;
            width: auto !important;
            transition: background 0.2s, border 0.2s;
        }
        section[data-testid="stFileUploader"] button:hover {
            background: #222 !important;
            border-color: #ff4d4d !important;
            color: #fff !important;
        }
        section[data-testid="stFileUploader"] * {
            background: transparent !important;
            color: #eeeeee !important;
            border: none !important;
        }
        section[data-testid="stFileUploader"] svg,
        section[data-testid="stFileUploader"] p {
            color: #eeeeee !important;
            fill: #eeeeee !important;
        }
        
        /* Success and Error messages */
        .stSuccess {
            background-color: #1a1a1a !important;
            border: 1px solid #2a4a2a !important;
            padding: 15px;
            border-radius: 8px;
        }
        .stError {
            background-color: #1a1a1a !important;
            border: 1px solid #4a2a2a !important;
            padding: 15px;
            border-radius: 8px;
        }
        
        /* Info boxes */
        .stInfo {
            background-color: #1a1a1a !important;
            border: 1px solid #333 !important;
            border-radius: 8px;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background-color: #333 !important;
        }
        .stProgress > div > div > div > div {
            background-color: #666 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: #121212 !important;
            border-bottom: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            color: #eeeeee !important;
            padding: 1rem 2rem;
            border-radius: 5px;
            background-color: #1e1e1e !important;
            border: 1px solid #333 !important;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #333 !important;
            border-color: #666 !important;
        }
        
        /* Dividers */
        hr {
            border-color: #333 !important;
        }
        
        /* Chat container */
        .chat-container {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            border: 1px solid #333;
        }
        
        /* Chat messages */
        .chat-message {
            background-color: #1a1a1a;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 8px;
            max-width: 80%;
            border: 1px solid #333;
        }
        
        /* Main container */
        .main .block-container {
            background-color: #1a1a1a !important;
            padding: 2rem;
            border-radius: 10px;
            border: 1px solid #333;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #1a1a1a !important;
        }
        
        /* Footer */
        footer {
            text-align: center;
            color: #aaa;
            margin-top: 40px;
            padding: 20px 0 10px 0;
            font-size: 1rem;
            border-top: 1px solid #333;
        }
        
        /* Title styling */
        .main-title {
            text-align: center !important;
            margin-bottom: 2rem !important;
            padding: 1rem 0 !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #eeeeee !important;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        
        /* Section headers */
        h2, h3 {
            text-align: left !important;
            margin: 1.5rem 0 1rem 0 !important;
            color: #eeeeee !important;
        }
        
        /* Description text */
        .description {
            text-align: left !important;
            margin-bottom: 2rem !important;
            color: #aaaaaa !important;
            font-size: 1.1rem !important;
        }
        
        /* Hide or recolor Streamlit top bar */
        header[data-testid="stHeader"] {
            background: #121212 !important;
        }
        .st-emotion-cache-18ni7ap {
            background: #121212 !important;
        }
        /* Hide the hamburger menu and status bar if desired */
        [data-testid="stToolbar"], .st-emotion-cache-1avcm0n {
            background: #121212 !important;
        }
        /* Remove blue shadow/underline if present */
        .stApp {
            box-shadow: none !important;
        }
        
        /* Streamlit selectbox dropdown menu styling */
        div[data-baseweb="select"] > div {
            background-color: #181818 !important;
            color: #eeeeee !important;
        }
        
        div[data-baseweb="select"] * {
            color: #eeeeee !important;
            background: #181818 !important;
        }

        div[data-baseweb="select"] [role="option"] {
            background: #181818 !important;
            color: #eeeeee !important;
        }
        
        div[data-baseweb="select"] [aria-selected="true"] {
            background: #222 !important;
            color: #ff4d4d !important;
        }

        /* Fix for the dropdown menu itself */
        div[data-baseweb="popover"] {
            background: #181818 !important;
            color: #eeeeee !important;
            border-radius: 8px !important;
            border: 1px solid #333 !important;
            z-index: 99999 !important;
        }

        div[data-baseweb="popover"] * {
            background: #181818 !important;
            color: #eeeeee !important;
        }

        div[data-baseweb="popover"] [role="option"] {
            background: #181818 !important;
            color: #eeeeee !important;
        }

        div[data-baseweb="popover"] [aria-selected="true"] {
            background: #222 !important;
            color: #ff4d4d !important;
        }

        /* General button styling for all buttons, including increment/decrement and dropdown arrows */
        button, .stButton>button, .stNumberInput button, .stSelectbox button, .stFileUploader button {
            background-color: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            transition: background 0.2s, border 0.2s;
        }

        /* Button hover effect */
        button:hover, .stButton>button:hover, .stNumberInput button:hover, .stSelectbox button:hover, .stFileUploader button:hover {
            background-color: #222 !important;
            border-color: #ff4d4d !important;
            color: #fff !important;
        }

        /* Remove white from selectbox dropdown arrow */
        .stSelectbox button[aria-label="Open dropdown"] {
            background: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
        }

        /* Remove white from number input increment/decrement buttons */
        .stNumberInput button {
            background-color: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
        }

        /* Remove white from file uploader button */
        .stFileUploader button {
            background: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
        }

        /* Remove white from Streamlit's default focus/active states */
        button:focus, .stButton>button:focus, .stNumberInput button:focus, .stSelectbox button:focus, .stFileUploader button:focus {
            outline: none !important;
            box-shadow: 0 0 0 2px #ff4d4d !important;
        }

        /* Make all number input fields black with white text */
        input[type="number"], .stNumberInput input {
            background-color: #181818 !important;
            color: #eeeeee !important;
            border: 1px solid #333 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }

        /* Make the increment/decrement buttons match the dark theme */
        .stNumberInput button {
            background-color: #181818 !important;
            color: #fff !important;
            border: 1.5px solid #333 !important;
            border-radius: 8px !important;
        }

        /* Remove white background on focus */
        input[type="number"]:focus, .stNumberInput input:focus {
            background-color: #222 !important;
            color: #fff !important;
            border-color: #ff4d4d !important;
            outline: none !important;
        }

        /* Hide increment and decrement (+/-) buttons from number inputs */
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
            display: none !important;
        }

        input[type="number"] {
            -moz-appearance: textfield; /* Firefox */
        }

        /* Hide Streamlit's custom number input buttons */
        .stNumberInput button {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load configuration
# Load configuration
try:
    with open("config.json", encoding="utf-8") as config_file:
        config_params = json.load(config_file)['params']
except Exception as e:
    st.error(f"Error loading config: {str(e)}")
    config = {
        "title": st.secrets["app_config"]["title"],
        "app_name": "Health Prediction App",
        "description": "Get personalized predictions for various health conditions using advanced machine learning models."
    }


# ------------------------- Constants -------------------------
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Model paths in models folder
MODEL_PATH = os.path.join('models', 'Diasease_model.h5')
BREAST_CANCER_MODEL_PATH = os.path.join('models', 'temp_model.h5')
DIABETES_MODEL_PATH = os.path.join('models', 'diabetes_model.pkl')
HEART_DISEASE_MODEL_PATH = os.path.join('models', 'heart_model.pkl')

# Google Drive URLs for models
# Google Drive URLs for models
BREAST_CANCER_MODEL_URL = st.secrets["model_urls"]["breast_cancer_model"]
DISEASE_MODEL_URL = st.secrets["model_urls"]["disease_model"]

# Google Drive URLs for models
BREAST_CANCER_MODEL_URL = "https://drive.google.com/file/d/1oUCacUPYAemX0zCJbRpVXutSjFtgac4K/view?usp=drive_link"
DISEASE_MODEL_URL = "https://drive.google.com/file/d/1h3RYnCvYNeE0HltyzVisU67DuTvaAdVf/view?usp=drive_link"


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

def download_model_from_gdrive(url, output_path):
    """Download a model file from Google Drive."""
    try:
        # Extract file ID from the URL
        if 'drive.google.com/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            file_id = url.split('/')[-2]
            
        download_url = f'https://drive.google.com/uc?id={file_id}'
        st.info(f"Attempting to download model from: {download_url}")
        
        # Try downloading with gdown
        gdown.download(download_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            st.success(f"Model downloaded successfully to {output_path}")
            return True
        else:
            st.error("Download completed but file not found at destination")
            return False
            
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        st.info("Please make sure the Google Drive file is publicly accessible with 'Anyone with the link' permission")
        return False

# Load intents from JSON file
def load_intents():
    try:
        with open('Intents.JSON', 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        st.warning(f"Error loading intents: {str(e)}. Using default intents.")
        return DEFAULT_INTENTS

# Default intents if Intents.JSON is not available
DEFAULT_INTENTS = {
    "intents": [
        {
            "patterns": ["hi", "hello", "hey", "greetings"],
            "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Greetings! How may I assist you?"]
        },
        {
            "patterns": ["diabetes", "diabetic", "blood sugar"],
            "responses": ["Diabetes is a condition that affects how your body processes blood sugar. Common symptoms include increased thirst, frequent urination, and fatigue. Would you like to know more about diabetes prediction?"]
        },
        {
            "patterns": ["heart", "cardiac", "chest pain"],
            "responses": ["Heart disease refers to several types of heart conditions. Common symptoms include chest pain, shortness of breath, and fatigue. Would you like to check your heart disease risk?"]
        },
        {
            "patterns": ["skin", "rash", "dermatology"],
            "responses": ["Skin conditions can range from minor to serious. Common issues include acne, eczema, and psoriasis. Would you like to analyze a skin condition?"]
        },
        {
            "patterns": ["breast", "mammogram", "cancer"],
            "responses": ["Breast cancer screening is important for early detection. Would you like to analyze a breast ultrasound image?"]
        },
        {
            "patterns": ["bye", "goodbye", "see you"],
            "responses": ["Goodbye! Take care of your health!", "See you later! Stay healthy!", "Bye! Remember to maintain a healthy lifestyle!"]
        }
    ]
}

# ------------------------- Helper Functions -------------------------
@st.cache_resource
def load_breast_cancer_model():
    try:
        # First try to load from local models folder
        if os.path.exists(BREAST_CANCER_MODEL_PATH):
            model = tf.keras.models.load_model(BREAST_CANCER_MODEL_PATH, compile=False)
            return model
        
        # If not found locally, download from Google Drive
        st.info("Downloading breast cancer model from Google Drive...")
        if download_model_from_gdrive(BREAST_CANCER_MODEL_URL, BREAST_CANCER_MODEL_PATH):
            model = tf.keras.models.load_model(BREAST_CANCER_MODEL_PATH, compile=False)
            st.success("Breast cancer model downloaded and loaded successfully!")
            return model
        else:
            st.error("Failed to download breast cancer model.")
            return None
            
    except Exception as e:
        st.error(f"Error loading breast cancer model: {str(e)}")
        return None

def predict_breast_cancer(image_data, model):
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = tf.image.resize(np.array(img), (256, 256))
        img_array = tf.expand_dims(img_array, 0) / 255.0
        predictions = model.predict(img_array)
        predicted_class = CLASS_MAPPING[np.argmax(predictions[0])]
        return predicted_class, predictions[0].tolist()
    except Exception as e:
        st.error(f"Error predicting breast cancer: {str(e)}")
        return None, None

def predict_skin_disease(image_data):
    try:
        # First try to load from local models folder
        if not os.path.exists(MODEL_PATH):
            st.info("Downloading disease model from Google Drive...")
            if not download_model_from_gdrive(DISEASE_MODEL_URL, MODEL_PATH):
                st.error("Failed to download disease model.")
                return None, None
        
        model = load_model(MODEL_PATH, compile=False)
        optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        probabilities = model.predict(img_array)
        prediction = np.argmax(probabilities, axis=1)[0]
        labels = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
        predicted_class = labels[prediction]
        
        return predicted_class, probabilities[0].tolist()
    except Exception as e:
        st.error(f"Error predicting skin disease: {str(e)}")
        return None, None

def predict_diabetes(features):
    try:
        if not os.path.exists(DIABETES_MODEL_PATH):
            st.error(f"Model file {DIABETES_MODEL_PATH} not found. Please upload the model to the models folder.")
            return None
        with open(DIABETES_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "🩺 You are a diabetic person. Regular monitoring and lifestyle strategies are essential."
            if prediction == 1 else
            "✅ Relax! You are not a diabetic person."
        )
    except Exception as e:
        st.error(f"Error predicting diabetes: {str(e)}")
        return None

def predict_heart_disease(features):
    try:
        if not os.path.exists(HEART_DISEASE_MODEL_PATH):
            st.error(f"Model file {HEART_DISEASE_MODEL_PATH} not found. Please upload the model to the models folder.")
            return None
        with open(HEART_DISEASE_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "💔 You have a heart problem. Seek medical advice immediately."
            if prediction == 1 else
            "❤️ Relax! You have a healthy heart."
        )
    except Exception as e:
        st.error(f"Error predicting heart disease: {str(e)}")
        return None

def get_chatbot_response(user_input):
    try:
        intents = load_intents()
        user_input = user_input.lower()
        
        for intent in intents['intents']:
            if any(pattern.lower() in user_input for pattern in intent['patterns']):
                return random.choice(intent['responses'])
        
        return "I'm not sure I understand. Could you please rephrase your question? I can help you with diabetes, heart disease, skin conditions, or breast cancer detection."
    except Exception as e:
        st.error(f"Error in chatbot response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again later."

# ------------------------- Main UI -------------------------
def main():
    # Header with centered title only
    st.markdown("<h1 class='main-title'>Health Prognosis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='description'>{config_params['description']}</p>", unsafe_allow_html=True)

    # --- Add Model Info Section ---
    st.markdown("""
    <div style='background-color: #181818; border-radius: 10px; padding: 20px; margin-bottom: 30px; border: 1px solid #222;'>
        <h3 style='color:#ff4d4d; margin-top:0;'>🧠 About the Models</h3>
        <ul style='color:#eee;'>
            <li><b>Breast Cancer Detection:</b> Deep Learning CNN model trained on ultrasound images. Classifies as Benign, Malignant, or Normal.</li>
            <li><b>Skin Disease Prediction:</b> Multi-class CNN model trained on dermatology images. Detects Acne, Melanoma, Psoriasis, Rosacea, and Vitiligo.</li>
            <li><b>Diabetes Prediction:</b> Machine Learning model (Random Forest) using clinical data (Pregnancies, Glucose, Blood Pressure, etc.).</li>
            <li><b>Heart Disease Prediction:</b> Machine Learning model (Logistic Regression) using patient health metrics (Age, Cholesterol, etc.).</li>
        </ul>
        <p style='color:#aaa; font-size:0.95rem; margin-top:10px;'>
            <b>Note:</b> These models are for educational and informational purposes. For medical advice, consult a healthcare professional.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chat Section
    st.markdown("### 🤖 Sam")
    st.markdown("Ask me health-related questions!")
    
    # Chat input
    user_input = st.text_input("Type your message...", key="chat_input")
    if user_input:
        response = get_chatbot_response(user_input)
        st.markdown(f"<div style='background-color: #333; padding: 10px; border-radius: 8px; margin-right: 20%;'>{response}</div>", unsafe_allow_html=True)

    # Breast Cancer Detection
    st.markdown("### Breast Cancer Detection 🔬")
    st.markdown("Upload an ultrasound image to predict whether the tumor is **Benign**, **Malignant**, or **Normal**.")
    
    uploaded_file = st.file_uploader("Choose an ultrasound image", type=["jpg", "jpeg", "png"], key="breast_cancer")
    if uploaded_file:
        image_data = uploaded_file.read()
        model = load_breast_cancer_model()
        if model:
            predicted_class, probabilities = predict_breast_cancer(image_data, model)
            if predicted_class:
                st.success(f"Prediction: {predicted_class}")
                st.write("Probabilities:")
                st.write(f"Benign: {probabilities[0]*100:.2f}%")
                st.write(f"Malignant: {probabilities[1]*100:.2f}%")
                st.write(f"Normal: {probabilities[2]*100:.2f}%")

    # Skin Disease Prediction
    st.markdown("### Skin Disease Prediction 🧴")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="skin_disease")
    if uploaded_file:
        image_data = uploaded_file.read()
        predicted_class, probabilities = predict_skin_disease(image_data)
        if predicted_class:
            st.success(f"Prediction: {predicted_class}")
            if probabilities:
                st.write("Probabilities:")
                labels = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
                for label, prob in zip(labels, probabilities):
                    st.write(f"{label}: {prob*100:.2f}%")

    # Diabetes Prediction
    st.markdown("### Diabetes Prediction 🥪")
    st.markdown("Fill out the details below to predict your risk of diabetes.")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose", min_value=0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=0)
    
    if st.button("Predict⚡️", key="predict_diabetes"):
        features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_diabetes(features)
        if result:
            st.success(result)

    # Heart Disease Prediction
    st.markdown("### Heart Disease Prediction ❤️")
    st.markdown("Fill out the details below to check your risk of heart disease.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, key="heart_age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="heart_sex")
        cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        chol = st.number_input("Cholesterol", min_value=0)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0)
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])
    
    if st.button("Predict⚡️", key="predict_heart"):
        sex_val = 1 if sex == "Male" else 0
        features = [age, sex_val, cp, trestbps, chol, fbs, thalach, exang, oldpeak, ca, thal]
        result = predict_heart_disease(features)
        if result:
            st.success(result)

    # Footer
    st.markdown("""
        <footer style='text-align:center; color:#aaa; margin-top:40px; padding:20px 0 10px 0; font-size:1rem;'>
            Created By Nitin Kharade | © 2025 All rights reserved.
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
