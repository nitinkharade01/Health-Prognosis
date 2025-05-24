from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random
import io
import base64

app = Flask(__name__)

# Load configuration
try:
    with open("config.json", encoding="utf-8") as config_file:
        config_params = json.load(config_file)['params']
except Exception as e:
    print(f"Error loading config: {str(e)}")
    config_params = {
        "app_name": "Health Prediction App",
        "description": "Get personalized predictions for various health conditions using advanced machine learning models."
    }

# Constants
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
MODEL_PATH = 'Diasease_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# Helper Functions
def load_breast_cancer_model():
    try:
        temp_model_path = "temp_model.h5"
        if os.path.exists(temp_model_path):
            model = tf.keras.models.load_model(temp_model_path, compile=False)
            return model
        print("Warning: temp_model.h5 not found. Please provide the model file locally.")
        return None
    except Exception as e:
        print(f"Error loading breast cancer model: {str(e)}")
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
        print(f"Error predicting breast cancer: {str(e)}")
        return None, None

def predict_skin_disease(image_data):
    try:
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
        print(f"Error predicting skin disease: {str(e)}")
        return None, None

def predict_diabetes(features):
    try:
        model_path = 'diabetes_model.pkl'
        if not os.path.exists(model_path):
            return "Model file not found. Please upload the model."
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "🩺 You are a diabetic person. Regular monitoring and lifestyle strategies are essential."
            if prediction == 1 else
            "✅ Relax! You are not a diabetic person."
        )
    except Exception as e:
        print(f"Error predicting diabetes: {str(e)}")
        return f"Error predicting diabetes: {str(e)}"

def predict_heart_disease(features):
    try:
        model_path = 'heart_model.pkl'
        if not os.path.exists(model_path):
            return "Model file not found. Please upload the model."
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "💔 You have a heart problem. Seek medical advice immediately."
            if prediction == 1 else
            "❤️ Relax! You have a healthy heart."
        )
    except Exception as e:
        print(f"Error predicting heart disease: {str(e)}")
        return f"Error predicting heart disease: {str(e)}"

def load_intents():
    try:
        if os.path.exists('Intents.JSON'):
            with open('Intents.JSON', 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():
                    return json.loads(content)
        return DEFAULT_INTENTS
    except Exception as e:
        print(f"Error loading intents: {str(e)}")
        return DEFAULT_INTENTS

def get_chatbot_response(user_input):
    try:
        intents = load_intents()
        user_input = user_input.lower()
        
        for intent in intents['intents']:
            if any(pattern.lower() in user_input for pattern in intent['patterns']):
                return random.choice(intent['responses'])
        
        return "I'm not sure I understand. Could you please rephrase your question? I can help you with diabetes, heart disease, skin conditions, or breast cancer detection."
    except Exception as e:
        print(f"Error in chatbot response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again later."

# Routes
@app.route('/')
def home():
    return render_template('index.html', config=config_params)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = get_chatbot_response(user_input)
    return jsonify({'response': response})

@app.route('/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer_route():
    try:
        file = request.files['file']
        image_data = file.read()
        model = load_breast_cancer_model()
        if model:
            predicted_class, probabilities = predict_breast_cancer(image_data, model)
            if predicted_class:
                return jsonify({
                    'prediction': predicted_class,
                    'probabilities': {
                        'Benign': f"{probabilities[0]*100:.2f}%",
                        'Malignant': f"{probabilities[1]*100:.2f}%",
                        'Normal': f"{probabilities[2]*100:.2f}%"
                    }
                })
        return jsonify({'error': 'Failed to make prediction'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/skin-disease', methods=['POST'])
def predict_skin_disease_route():
    try:
        file = request.files['file']
        image_data = file.read()
        predicted_class, probabilities = predict_skin_disease(image_data)
        if predicted_class:
            labels = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
            prob_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(labels, probabilities)}
            return jsonify({
                'prediction': predicted_class,
                'probabilities': prob_dict
            })
        return jsonify({'error': 'Failed to make prediction'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes_route():
    try:
        data = request.json
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['dpf']),
            float(data['age'])
        ]
        result = predict_diabetes(features)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease_route():
    try:
        data = request.json
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            float(data['fbs']),
            float(data['thalach']),
            float(data['exang']),
            float(data['oldpeak']),
            float(data['ca']),
            float(data['thal'])
        ]
        result = predict_heart_disease(features)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)