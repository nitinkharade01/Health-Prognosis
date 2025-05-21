from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pickle
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import io
import base64

app = Flask(__name__)

# Load configuration
with open("config.json", encoding="utf-8") as config_file:
    config_params = json.load(config_file)['params']

# Constants
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
MODEL_PATH = 'Diasease_model.h5'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper Functions
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
            }
        ]
    }
    
    try:
        with open('prompt_template.json', 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        return False

def load_chatbot_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        intents = None
        prompt_template = None
        
        try:
            if os.path.exists('Intents.JSON'):
                with open('Intents.JSON', 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip():
                        intents = json.loads(content)
        except Exception as e:
            print(f"Error reading Intents.JSON: {str(e)}")
            
        try:
            if not os.path.exists('prompt_template.json'):
                create_prompt_template()
            
            with open('prompt_template.json', 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():
                    prompt_template = json.loads(content)
        except Exception as e:
            print(f"Error reading prompt_template.json: {str(e)}")
            
        return model, tokenizer, intents, prompt_template
    except Exception as e:
        print(f"Error loading chatbot model: {str(e)}")
        return None, None, None, None

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
        return f"Error generating response: {str(e)}"

def get_chatbot_response(user_input):
    try:
        model, tokenizer, intents, prompt_template = load_chatbot_model()
        
        if not all([model, tokenizer, intents, prompt_template]):
            return "I apologize, but I'm currently unable to process your request due to configuration issues."
        
        for intent in intents['intents']:
            if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                return random.choice(intent['responses'])
        
        try:
            system_prompt = prompt_template['system_prompt']
            prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
            response = generate_gpt2_response(prompt, model, tokenizer)
            response = response.replace(prompt, "").strip()
            return response
        except Exception as api_error:
            for intent in intents['intents']:
                if any(pattern.lower() in user_input.lower() for pattern in intent['patterns']):
                    return random.choice(intent['responses'])
            return "I apologize, but I'm having trouble processing your request right now. Please try again later or rephrase your question."
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(file)
        model = load_breast_cancer_model()
        predicted_class, predictions = predict_breast_cancer(image, model)
        
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
        return jsonify({'error': str(e)}), 500

@app.route('/predict/skin-disease', methods=['POST'])
def predict_skin_disease_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        prediction = predict_skin_disease(file_path)
        os.remove(file_path)  # Clean up
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes_route():
    try:
        features = request.json.get('features', [])
        if len(features) != 8:
            return jsonify({'error': 'Invalid number of features'}), 400
        
        result = predict_diabetes(features)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease_route():
    try:
        features = request.json.get('features', [])
        if len(features) != 11:
            return jsonify({'error': 'Invalid number of features'}), 400
        
        result = predict_heart_disease(features)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)