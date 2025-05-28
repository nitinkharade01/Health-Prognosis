from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pickle
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False
    print("Warning: tensorflow_addons not available. Some features may be limited.")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import io
import base64
import traceback
import joblib

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

# --- GLOBALS FOR CHATBOT ---
CHATBOT_MODEL = None
CHATBOT_TOKENIZER = None
CHATBOT_INTENTS = None
CHATBOT_PROMPT_TEMPLATE = None

# --- Load Chatbot Model and Data Once ---
def initialize_chatbot():
    global CHATBOT_MODEL, CHATBOT_TOKENIZER, CHATBOT_INTENTS, CHATBOT_PROMPT_TEMPLATE
    try:
        if CHATBOT_MODEL is None or CHATBOT_TOKENIZER is None:
            CHATBOT_TOKENIZER = AutoTokenizer.from_pretrained('gpt2')
            CHATBOT_MODEL = AutoModelForCausalLM.from_pretrained('gpt2')
        if CHATBOT_INTENTS is None:
            if os.path.exists('Intents.JSON'):
                with open('Intents.JSON', 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip():
                        CHATBOT_INTENTS = json.loads(content)
        if CHATBOT_PROMPT_TEMPLATE is None:
            if not os.path.exists('prompt_template.json'):
                create_prompt_template()
            with open('prompt_template.json', 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():
                    CHATBOT_PROMPT_TEMPLATE = json.loads(content)
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")

# Call this once at startup
initialize_chatbot()

# Helper Functions
def load_breast_cancer_model():
    try:
        # Try to load from local file first
        temp_model_path = "temp_model.h5"
        if os.path.exists(temp_model_path):
            model = tf.keras.models.load_model(temp_model_path, compile=False)
            return model
        # If not found, fallback to old logic (but warn user)
        print("Warning: temp_model.h5 not found. Please provide the model file locally.")
        return None
    except Exception as e:
        print(f"Error loading breast cancer model: {str(e)}")
        return None

def predict_breast_cancer(image: Image.Image, model):
    """Preprocess image and predict class and probabilities."""
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = CLASS_MAPPING[np.argmax(predictions[0])]
    return predicted_class, predictions

def predict_skin_disease(file_path):
    try:
        model = load_model(MODEL_PATH, compile=False)
        optimizer = tf.keras.optimizers.AdamW(weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        labels = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        probabilities = model.predict(img_array)
        prediction = np.argmax(probabilities, axis=1)[0]
        predicted_class = labels[prediction]

        return predicted_class, probabilities
    except Exception as e:
        print(f"Error predicting skin disease: {str(e)}")
        return None

def predict_diabetes(features):
    try:
        model_path = 'diabetes_model.pkl'
        if not os.path.exists(model_path):
            error_msg = f"Model file {model_path} not found. Please upload the model."
            print(error_msg)
            return error_msg
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "🦪 You are a diabetic person. Regular monitoring and lifestyle strategies are essential."
            if prediction == 1 else
            "✅ Relax! You are not a diabetic person."
        )
    except ValueError as ve:
        if 'incompatible dtype' in str(ve):
            msg = ("Model file is incompatible with the current scikit-learn version. "
                   "Please re-train and re-save the model using your current environment, "
                   "or ensure you are using the same scikit-learn version as when the model was created.")
            print(msg)
            return msg
        print(f"Error predicting diabetes: {type(ve).__name__}: {str(ve)}")
        return f"Error predicting diabetes: {type(ve).__name__}: {str(ve)}"
    except Exception as e:
        print(f"Error predicting diabetes: {type(e).__name__}: {str(e)}")
        return f"Error predicting diabetes: {type(e).__name__}: {str(e)}"

def predict_heart_disease(features):
    try:
        model_path = 'heart_model.pkl'
        if not os.path.exists(model_path):
            error_msg = f"Model file {model_path} not found. Please upload the model."
            print(error_msg)
            return error_msg
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        prediction = model.predict([features])[0]
        return (
            "💔 You have a heart problem. Seek medical advice immediately."
            if prediction == 1 else
            "❤️ Relax! You have a healthy heart."
        )
    except ValueError as ve:
        if 'incompatible dtype' in str(ve):
            msg = ("Model file is incompatible with the current scikit-learn version. "
                   "Please re-train and re-save the model using your current environment, "
                   "or ensure you are using the same scikit-learn version as when the model was created.")
            print(msg)
            return msg
        print(f"Error predicting heart disease: {type(ve).__name__}: {str(ve)}")
        return f"Error predicting heart disease: {type(ve).__name__}: {str(ve)}"
    except Exception as e:
        print(f"Error predicting heart disease: {type(e).__name__}: {str(e)}")
        return f"Error predicting heart disease: {type(e).__name__}: {str(e)}"

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
            }
        ]
    }
    
    try:
        with open('prompt_template.json', 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error creating prompt template: {str(e)}")
        return False

def generate_gpt2_response(prompt, model, tokenizer, max_length=100):
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
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
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating GPT-2 response: {str(e)}")
        return None

def get_chatbot_response(user_input):
    try:
        # Use global loaded models/data
        model = CHATBOT_MODEL
        tokenizer = CHATBOT_TOKENIZER
        intents = CHATBOT_INTENTS
        prompt_template = CHATBOT_PROMPT_TEMPLATE
        if not all([model, tokenizer, intents, prompt_template]):
            return "I apologize, but I'm currently unable to process your request due to configuration issues."
        for intent in intents['intents']:
            if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                return random.choice(intent['responses'])
        try:
            system_prompt = prompt_template['system_prompt']
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
            response = generate_gpt2_response(prompt, model, tokenizer)
            if response:
                response = response.replace(prompt, "").strip()
                return response
            return "I apologize, but I'm having trouble generating a response right now."
        except Exception as api_error:
            for intent in intents['intents']:
                if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                    return random.choice(intent['responses'])
            return "I apologize, but I'm having trouble processing your request right now. Please try again later or rephrase your question."
    except Exception as e:
        print(f"Error in chatbot response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again later."

# Routes
@app.route('/')
def home():
    return render_template('index.html', config=config_params)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        response = get_chatbot_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Ensure image is in RGB mode
        image = Image.open(file).convert('RGB')
        model = load_breast_cancer_model()
        if not model:
            return jsonify({'error': 'Failed to load model'}), 500
            
        predicted_class, predictions = predict_breast_cancer(image, model)
        if predicted_class is None or predictions is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        if isinstance(predicted_class, str) and (predicted_class.startswith('Error') or 'incompatible' in predicted_class):
            return jsonify({'error': predicted_class}), 500
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': float(np.max(predictions[0])),
            'probabilities': {
                'benign': float(predictions[0][0]),
                'malignant': float(predictions[0][1]),
                'normal': float(predictions[0][2])
            }
        })
    except Exception as e:
        import sys
        tb_str = traceback.format_exc()
        print(f"Error in breast cancer prediction: {str(e)}\n{tb_str}")
        # Also log to a file for persistent debugging
        with open('breast_cancer_error.log', 'a', encoding='utf-8') as log_file:
            log_file.write(f"Error: {str(e)}\nType: {type(e).__name__}\nTraceback:\n{tb_str}\n---\n")
        return jsonify({'error': f'Error processing the image: {type(e).__name__}: {str(e)}'}), 500

@app.route('/predict/skin-disease', methods=['POST'])
def predict_skin_disease_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        # Ensure image is in RGB mode before saving
        img = Image.open(file).convert('RGB')
        img.save(file_path)
        prediction = predict_skin_disease(file_path)
        os.remove(file_path)  # Clean up
        
        if not prediction:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        # Assuming predict_skin_disease returns a tuple (predicted_class, probabilities)
        predicted_class, probabilities = prediction
        confidence = float(np.max(probabilities[0]))
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'Acne': float(probabilities[0][0]),
                'Melanoma': float(probabilities[0][1]),
                'Psoriasis': float(probabilities[0][2]),
                'Rosacea': float(probabilities[0][3]),
                'Vitiligo': float(probabilities[0][4])
            }
        })
    except Exception as e:
        print(f"Error in skin disease prediction: {str(e)}")
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes_route():
    try:
        features = request.json.get('features', [])
        if len(features) != 8:
            return jsonify({'error': 'Invalid number of features'}), 400
        result = predict_diabetes(features)
        if not result:
            return jsonify({'error': 'Failed to make prediction'}), 500
        if isinstance(result, str) and (result.startswith('Error') or 'incompatible' in result):
            return jsonify({'error': result}), 500
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Error in diabetes prediction: {str(e)}")
        return jsonify({'error': 'Error making prediction'}), 500

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease_route():
    try:
        features = request.json.get('features', [])
        if len(features) != 11:
            return jsonify({'error': 'Invalid number of features'}), 400
        result = predict_heart_disease(features)
        if not result:
            return jsonify({'error': 'Failed to make prediction'}), 500
        if isinstance(result, str) and (result.startswith('Error') or 'incompatible' in result):
            return jsonify({'error': result}), 500
        return jsonify({'prediction': result})
    except Exception as e:
        print(f"Error in heart disease prediction: {str(e)}")
        return jsonify({'error': 'Error making prediction'}), 500

@app.route('/check_model', methods=['GET'])
def check_model():
    model_path = "temp_model.h5"
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return jsonify({"status": "success", "message": "Model uploaded successfully", "model_path": model_path, "size_bytes": size_bytes})
    else:
         return jsonify({"status": "error", "message": "Model file (temp_model.h5) not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)