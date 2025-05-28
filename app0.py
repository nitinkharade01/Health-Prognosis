# ------------------------- Imports -------------------------
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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

# ------------------------- Page Configuration -------------------------
with open("config.json", encoding="utf-8") as config_file:
    config_params = json.load(config_file)['params']

# ------------------------- UI Title -------------------------
st.title(config_params["app_name"])

# ------------------------- Custom Dark Theme Styling -------------------------
st.markdown("""
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #121212;
            color: #eeeeee;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #f0f0f0 !important;
        }

        /* Text elements */
        p, span, label, div {
            color: #eeeeee !important;
        }

        /* Input elements */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            background-color: #222222 !important;
            color: #ffffff !important;
            border: 1px solid #444 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
            border-radius: 4px !important;
            padding: 0.3em 0.6em !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            text-transform: capitalize !important;
            letter-spacing: 0.02em !important;
            transition: background-color 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #111111 !important;
        }

        /* File uploader */
        .stFileUploader {
            background-color: #222222 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            padding: 1em !important;
        }

        .stFileUploader > div > div > div {
            background-color: #333333 !important;
            color: #ffffff !important;
        }

        .stFileUploader > div > div > div > div {
            background-color: #222222 !important;
            color: #ffffff !important;
        }

        .stFileUploader > div > div > div > div > div {
            background-color: #333333 !important;
            color: #ffffff !important;
        }

        .stFileUploader > div > div > div > div > div > div {
            background-color: #222222 !important;
            color: #ffffff !important;
        }

        .stFileUploader > div > div > div > div > div > div > div {
            background-color: #333333 !important;
            color: #ffffff !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-track {
            background: #222222;
        }

        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #666;
        }

        /* Prediction sections */
        div[data-testid="stVerticalBlock"] > div:has(h3:contains("Diabetes Prediction")),
        div[data-testid="stVerticalBlock"] > div:has(h3:contains("Heart Disease Prediction")) {
            background-color: #1e1e1e !important;
            padding: 20px !important;
            border-radius: 12px !important;
            margin: 10px 0 !important;
        }

        /* Select boxes and dropdowns */
        .stSelectbox > div > div > div {
            background-color: #222222 !important;
            color: #ffffff !important;
        }

        /* Success messages */
        .stSuccess {
            background-color: #1e3a1e !important;
            color: #ffffff !important;
        }

        /* Error messages */
        .stError {
            background-color: #3a1e1e !important;
            color: #ffffff !important;
        }

        /* Info messages */
        .stInfo {
            background-color: #1e1e3a !important;
            color: #ffffff !important;
        }

        /* Chat interface */
        .stTextArea > div > div > textarea {
            background-color: #222222 !important;
            color: #ffffff !important;
        }

        /* Number inputs */
        .stNumberInput > div > div > input {
            background-color: #222222 !important;
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------- Constants -------------------------
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
MODEL_PATH = 'Diasease_model.h5'

# Try to get API key from Streamlit secrets first, then environment variable
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv('OPENAI_API_KEY'))

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
        "🦪 You are a diabetic person. Regular monitoring and lifestyle strategies are essential."
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

def create_prompt_template():
    template = {
        "system_prompt": "You are a knowledgeable and empathetic health assistant. Your role is to provide accurate, helpful, and supportive information about health and wellness topics. Always maintain a professional tone while being accessible and clear in your explanations.",
        
        "few_shot_examples": [
            {
                "user": "What are the symptoms of diabetes?",
                "assistant": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, and slow-healing sores. However, it's important to note that symptoms can vary between individuals. Would you like to know more about any specific symptom or about diabetes management?"
            },
            {
                "user": "How can I improve my sleep quality?",
                "assistant": "To improve sleep quality, try these strategies: maintain a consistent sleep schedule, create a relaxing bedtime routine, keep your bedroom cool and dark, limit screen time before bed, avoid caffeine and large meals close to bedtime, and exercise regularly but not too close to sleep. Would you like more specific tips about any of these areas?"
            },
            {
                "user": "What are some ways to manage stress?",
                "assistant": "Effective stress management techniques include: regular exercise, mindfulness meditation, deep breathing exercises, maintaining a healthy diet, getting adequate sleep, practicing time management, and engaging in hobbies or activities you enjoy. Would you like to learn more about any of these techniques?"
            }
        ],
        
        "response_templates": {
            "general_health": "When it comes to {topic}, it's important to {key_point}. Some key considerations include {considerations}. Would you like to know more about {specific_aspect}?",
            "medical_advice": "Regarding {condition}, common approaches include {approaches}. It's essential to {important_point}. Would you like more information about {specific_topic}?",
            "preventive_care": "For preventing {condition}, recommended measures include {measures}. Regular {preventive_action} is also important. Would you like to learn more about {specific_prevention}?"
        },
        
        "safety_disclaimers": [
            "This information is for educational purposes only and should not replace professional medical advice.",
            "Always consult with a healthcare provider for personalized medical advice.",
            "In case of emergency, please seek immediate medical attention."
        ],
        
        "follow_up_questions": [
            "Would you like more specific information about {topic}?",
            "Is there a particular aspect of {topic} you'd like to explore further?",
            "Would you like to know more about how to {action}?"
        ],
        
        "specialized_topics": {
            "nutrition": "When discussing nutrition, focus on balanced diets, portion control, and healthy eating habits.",
            "exercise": "For exercise topics, emphasize safety, proper form, and gradual progression.",
            "mental_health": "When addressing mental health, maintain sensitivity and encourage professional support when needed."
        },
        
        "emergency_guidelines": {
            "recognize_emergency": "If you're experiencing {symptoms}, seek immediate medical attention.",
            "first_aid": "For {situation}, follow these steps: {steps}. Then seek medical help.",
            "preventive_measures": "To prevent {condition}, consider these measures: {measures}."
        }
    }
    
    try:
        with open('prompt_template.json', 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error creating prompt template: {str(e)}")
        return False

@st.cache_resource
def load_chatbot_model():
    try:
        # Load GPT-2 model and tokenizer using Auto classes
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        # Load intents and prompt template with proper error handling
        intents = None
        prompt_template = None
        
        # Try to load Intents.JSON
        try:
            if os.path.exists('Intents.JSON'):
                with open('Intents.JSON', 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip():  # Check if file is not empty
                        intents = json.loads(content)
                    else:
                        st.error("Intents.JSON is empty")
            else:
                st.error("Intents.JSON file not found")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Intents.JSON: {str(e)}")
        except Exception as e:
            st.error(f"Error reading Intents.JSON: {str(e)}")
            
        # Try to load or create prompt_template.json
        try:
            if not os.path.exists('prompt_template.json'):
                if not create_prompt_template():
                    st.error("Failed to create prompt template file")
                    return None, None, None, None
            
            with open('prompt_template.json', 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():  # Check if file is not empty
                    prompt_template = json.loads(content)
                else:
                    st.error("prompt_template.json is empty")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing prompt_template.json: {str(e)}")
            # Try to recreate the file if parsing fails
            if create_prompt_template():
                try:
                    with open('prompt_template.json', 'r', encoding='utf-8') as file:
                        prompt_template = json.load(file)
                except Exception as e:
                    st.error(f"Error reading recreated prompt template: {str(e)}")
        except Exception as e:
            st.error(f"Error reading prompt_template.json: {str(e)}")
            
        if intents is None or prompt_template is None:
            st.error("Failed to load required configuration files")
            return None, None, None, None
            
        return model, tokenizer, intents, prompt_template
    except Exception as e:
        st.error(f"Error loading chatbot model: {str(e)}")
        return None, None, None, None

def generate_gpt2_response(prompt, model, tokenizer, max_length=100):
    try:
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode and return the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_chatbot_response(user_input):
    try:
        model, tokenizer, intents, prompt_template = load_chatbot_model()
        
        if not all([model, tokenizer, intents, prompt_template]):
            return "I apologize, but I'm currently unable to process your request due to configuration issues. Please check the error message above."
        
        # First, try to match with intents
        for intent in intents['intents']:
            if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                return random.choice(intent['responses'])
        
        # If no intent matches, use GPT-2
        try:
            # Prepare the prompt using the template
            system_prompt = prompt_template['system_prompt']
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
            
            # Generate response using GPT-2
            response = generate_gpt2_response(prompt, model, tokenizer)
            
            # Clean up the response
            response = response.replace(prompt, "").strip()
            return response
        except Exception as api_error:
            # For any errors, try to provide a helpful response based on the intent matching
            for intent in intents['intents']:
                if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                    return random.choice(intent['responses'])
            return "I apologize, but I'm having trouble processing your request right now. Please try again later or rephrase your question."
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# ------------------------- Home Page -------------------------
st.header("Welcome to the Health Prediction App")
st.markdown(f"""
    <div style="font-size:18px;">{config_params['description']}</div>
    <div style="font-size:16px; color: gray;">
        Get personalized predictions for skin diseases, diabetes, and heart health using advanced machine learning models.
    </div>
""", unsafe_allow_html=True)

# ------------------------- Chatbot -------------------------
st.header("🤖 Sam")
st.markdown("Ask me health-related questions!")
user_input = st.text_input("You:", "", key="chat_input")
if user_input:
    response = get_chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=150, key="chatbot_response")

# ------------------------- Breast Cancer Detection -------------------------
st.header("Breast Cancer Detection 🔬")
st.markdown("Upload an ultrasound image to predict whether the tumor is **Benign**, **Malignant**, or **Normal**.")

uploaded_file = st.file_uploader("Choose an ultrasound image", type=["jpg", "jpeg", "png"], key="bc_file")
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

# ------------------------- Skin Disease Prediction -------------------------
st.header("Skin Disease Prediction 🧴")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="skin_file")
if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join('uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    prediction = predict_skin_disease(file_path)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {prediction}")

# ------------------------- Diabetes Prediction -------------------------
st.header("Diabetes Prediction 🥪")
st.markdown("Fill out the details below to predict your risk of diabetes.")

pregnancies = st.number_input("Pregnancies", min_value=0, key="diabetes_pregnancies")
glucose = st.number_input("Glucose", min_value=0, key="diabetes_glucose")
blood_pressure = st.number_input("Blood Pressure", min_value=0, key="diabetes_blood_pressure")
skin_thickness = st.number_input("Skin Thickness", min_value=0, key="diabetes_skin_thickness")
insulin = st.number_input("Insulin", min_value=0, key="diabetes_insulin")
bmi = st.number_input("BMI", min_value=0.0, key="diabetes_bmi")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, key="diabetes_dpf")
age = st.number_input("Age", min_value=0, key="diabetes_age")

features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
if st.button("Predict⚡️", key="diabetes_predict"):
    result = predict_diabetes(features)
    st.success(result)

# ------------------------- Heart Disease Prediction -------------------------
st.header("Heart Disease Prediction ❤️")
st.markdown("Fill out the details below to check your risk of heart disease.")

age = st.number_input("Age", min_value=0, key="heart_age")
sex = st.selectbox("Sex", ["Male", "Female"], key="heart_sex")
cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, key="heart_cp")
trestbps = st.number_input("Resting Blood Pressure", min_value=0, key="heart_trestbps")
chol = st.number_input("Cholesterol", min_value=0, key="heart_chol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], key="heart_fbs")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, key="heart_thalach")
exang = st.selectbox("Exercise Induced Angina", [0, 1], key="heart_exang")
oldpeak = st.number_input("ST Depression", min_value=0.0, key="heart_oldpeak")
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3], key="heart_ca")
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3], key="heart_thal")

sex_val = 1 if sex == "Male" else 0
features = [age, sex_val, cp, trestbps, chol, fbs, thalach, exang, oldpeak, ca, thal]

if st.button("Predict⚡️", key="heart_predict"):
    result = predict_heart_disease(features)
    st.success(result)
