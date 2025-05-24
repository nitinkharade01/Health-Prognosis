import streamlit as st
import os
import json
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
import random
import io
import gdown
import sqlite3
import bcrypt
import secrets
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ------------------------- Constants -------------------------
CLASS_MAPPING = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Model paths in models folder
MODEL_PATH = os.path.join('models', 'Diasease_model.h5')
BREAST_CANCER_MODEL_PATH = os.path.join('models', 'temp_model.h5')
DIABETES_MODEL_PATH = os.path.join('models', 'diabetes_model.pkl')
HEART_DISEASE_MODEL_PATH = os.path.join('models', 'heart_model.pkl')

# Google Drive URLs for models
BREAST_CANCER_MODEL_URL = st.secrets["model_urls"]["breast_cancer_model"]
DISEASE_MODEL_URL = st.secrets["model_urls"]["disease_model"]

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# ------------------------- Model Functions -------------------------
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

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="Health Prognosis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------- Database Setup -------------------------
DB_PATH = 'secure_users.db'

def init_db():
    """Initialize the SQLite database with secure tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table with secure fields
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  salt TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_login TIMESTAMP,
                  is_active BOOLEAN DEFAULT 1,
                  is_admin BOOLEAN DEFAULT 0)''')
    
    # Create login attempts table for security
    c.execute('''CREATE TABLE IF NOT EXISTS login_attempts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  ip_address TEXT,
                  success BOOLEAN)''')
    
    # Create password reset table
    c.execute('''CREATE TABLE IF NOT EXISTS password_resets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT NOT NULL,
                  token TEXT NOT NULL,
                  expiry TIMESTAMP NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    """Hash a password using bcrypt with a salt."""
    if salt is None:
        salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt), salt

def verify_password(password, password_hash, salt):
    """Verify a password against its hash and salt."""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash)

def add_user(username, password, email, is_admin=False):
    """Add a new user with secure password hashing."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if username or email already exists
        c.execute('SELECT username, email FROM users WHERE username = ? OR email = ?',
                 (username, email))
        existing = c.fetchone()
        if existing:
            if existing[0] == username:
                return False, "Username already exists"
            return False, "Email already registered"
        
        # Hash password with salt
        password_hash, salt = hash_password(password)
        
        # Insert new user
        c.execute('''INSERT INTO users 
                    (username, password_hash, email, salt, is_admin)
                    VALUES (?, ?, ?, ?, ?)''',
                 (username, password_hash, email, salt, is_admin))
        
        conn.commit()
        return True, "User registered successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def verify_user(username, password):
    """Verify user credentials with secure password checking."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get user data
        c.execute('''SELECT id, username, password_hash, salt, is_admin, is_active 
                    FROM users WHERE username = ?''', (username,))
        user = c.fetchone()
        
        if not user:
            return False, None
        
        # Check if account is active
        if not user[5]:  # is_active
            return False, "Account is deactivated"
        
        # Verify password
        if verify_password(password, user[2], user[3]):  # password_hash, salt
            # Update last login
            c.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                     (user[0],))
            conn.commit()
            return True, {"id": user[0], "username": user[1], "is_admin": user[4]}
        
        return False, "Invalid password"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def get_user_info(user_id):
    """Get user information (excluding sensitive data)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT username, email, created_at, last_login, is_admin 
                    FROM users WHERE id = ?''', (user_id,))
        return c.fetchone()
    except Exception as e:
        return None
    finally:
        conn.close()

def get_all_users():
    """Get all users (for admin panel, excluding sensitive data)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('''
            SELECT id, username, email, created_at, last_login, is_admin, is_active 
            FROM users
        ''', conn)
        return df
    except Exception as e:
        return pd.DataFrame()
    finally:
        conn.close()

# ------------------------- Session State -------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# ------------------------- Login/Register UI -------------------------
def generate_reset_token():
    """Generate a secure reset token."""
    return secrets.token_urlsafe(32)

def store_reset_token(username, token):
    """Store reset token in database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        expiry = datetime.now() + timedelta(hours=1)  # Token expires in 1 hour
        c.execute('''INSERT OR REPLACE INTO password_resets 
                    (username, token, expiry) VALUES (?, ?, ?)''',
                 (username, token, expiry))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error storing reset token: {str(e)}")
        return False
    finally:
        conn.close()

def verify_reset_token(username, token):
    """Verify if reset token is valid."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT expiry FROM password_resets 
                    WHERE username = ? AND token = ?''',
                 (username, token))
        result = c.fetchone()
        
        if result and datetime.now() < datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S.%f'):
            return True
        return False
    except Exception as e:
        st.error(f"Error verifying reset token: {str(e)}")
        return False
    finally:
        conn.close()

def reset_password(username, new_password):
    """Reset user's password."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Hash new password
        password_hash, salt = hash_password(new_password)
        
        # Update password
        c.execute('''UPDATE users 
                    SET password_hash = ?, salt = ? 
                    WHERE username = ?''',
                 (password_hash, salt, username))
        
        # Remove used token
        c.execute('DELETE FROM password_resets WHERE username = ?', (username,))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error resetting password: {str(e)}")
        return False
    finally:
        conn.close()

def send_reset_email(email, username, token):
    """Send password reset email."""
    try:
        # Email configuration from secrets
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = st.secrets["email"]["smtp_port"]
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = "Password Reset - Health Prognosis"

        # Create email body
        body = f"""
        Hello {username},

        You have requested to reset your password for the Health Prognosis application.

        Your reset token is: {token}

        To reset your password:
        1. Go to the Health Prognosis application
        2. Click on the "Reset Password" tab
        3. Enter your username and this token
        4. Enter your new password

        This token will expire in 1 hour.

        If you did not request this password reset, please ignore this email.

        Best regards,
        Health Prognosis Team
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def login_page():
    st.markdown("<h1 class='main-title'>Health Prognosis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Reset Password"])
    
    with tab1:
        st.markdown("### Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if username and password:
                success, result = verify_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_info = result
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error(result)
            else:
                st.error("Please enter both username and password")
    
    with tab2:
        st.markdown("### Register")
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        email = st.text_input("Email", key="register_email")
        
        if st.button("Register"):
            if not all([new_username, new_password, confirm_password, email]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long!")
            elif not email or '@' not in email:
                st.error("Please enter a valid email address!")
            else:
                success, message = add_user(new_username, new_password, email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    with tab3:
        st.markdown("### Reset Password")
        reset_username = st.text_input("Username", key="reset_username")
        reset_email = st.text_input("Email", key="reset_email")
        
        if st.button("Request Reset"):
            if not all([reset_username, reset_email]):
                st.error("Please enter both username and email")
            else:
                # Verify username and email match
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute('SELECT email FROM users WHERE username = ?', (reset_username,))
                result = c.fetchone()
                conn.close()
                
                if result and result[0] == reset_email:
                    # Generate and store reset token
                    token = generate_reset_token()
                    if store_reset_token(reset_username, token):
                        # Send reset email
                        if send_reset_email(reset_email, reset_username, token):
                            st.success("Password reset instructions have been sent to your email.")
                        else:
                            st.error("Failed to send reset email. Please try again later.")
                    else:
                        st.error("Failed to generate reset token")
                else:
                    st.error("Username and email do not match")
        
        # Password reset form
        st.markdown("### Enter Reset Token")
        reset_token = st.text_input("Reset Token", key="reset_token")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new_password")
        
        if st.button("Reset Password"):
            if not all([reset_username, reset_token, new_password, confirm_password]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match!")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long!")
            elif not verify_reset_token(reset_username, reset_token):
                st.error("Invalid or expired reset token")
            else:
                if reset_password(reset_username, new_password):
                    st.success("Password has been reset successfully!")
                else:
                    st.error("Failed to reset password")

# ------------------------- Admin Panel -------------------------
def admin_panel():
    st.markdown("### Admin Panel")
    users_df = get_all_users()
    if not users_df.empty:
        st.dataframe(users_df)
        
        # Export to Excel option
        if st.button("Export Users to Excel"):
            users_df.to_excel("users_export.xlsx", index=False)
            st.success("Users exported successfully!")
    else:
        st.warning("No users found in the database.")

# ------------------------- Main App -------------------------
def main_app():
    # Header with user info
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1 class='main-title'>Health Prognosis</h1>", unsafe_allow_html=True)
    with col2:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.experimental_rerun()
    
    user_info = get_user_info(st.session_state.user_info['id'])
    if user_info:
        st.markdown(f"<p class='description'>Welcome, {user_info[0]}!</p>", unsafe_allow_html=True)
    
    # Show admin panel if user is admin
    if st.session_state.user_info['is_admin']:
        admin_panel()

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

# ------------------------- Initialize Database -------------------------
init_db()

# ------------------------- Main Flow -------------------------
if not st.session_state.authenticated:
    login_page()
else:
    main_app()

# Add your custom CSS here (copy from your original app)
st.markdown("""
    <style>
        /* Your existing CSS styles */
    </style>
""", unsafe_allow_html=True) 