# Health Prognosis - AI-Powered Health Prediction System

A comprehensive health prediction system that uses machine learning to predict various health conditions including diabetes, heart disease, breast cancer, and skin diseases.

## Features

- **Breast Cancer Detection**: Deep Learning CNN model for ultrasound image analysis
- **Skin Disease Prediction**: Multi-class CNN model for dermatology image analysis
- **Diabetes Prediction**: Machine Learning model using clinical data
- **Heart Disease Prediction**: Machine Learning model using patient health metrics
- **Interactive Chatbot**: AI-powered health assistant for answering health-related queries

## Project Structure

```
health-prognosis/
├── models/                  # Model files (not tracked in git)
├── static/                  # Static files (CSS, images)
├── templates/              # HTML templates
├── uploads/                # Temporary upload directory
├── .streamlit/            # Streamlit configuration
├── config.json            # Application configuration
├── Intents.JSON           # Chatbot intents
├── requirements.txt       # Python dependencies
├── streamlit_app.py       # Main Streamlit application
└── README.md             # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/health-prognosis.git
cd health-prognosis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required models:
   - The application will automatically download the required models on first run
   - Models are downloaded from Google Drive and stored in the `models/` directory

## Running the Application

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## Model Information

- **Breast Cancer Model**: CNN trained on ultrasound images
- **Skin Disease Model**: CNN trained on dermatology images
- **Diabetes Model**: Random Forest classifier
- **Heart Disease Model**: Logistic Regression classifier

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- All models are for educational and informational purposes
- For medical advice, please consult healthcare professionals
- Created by Nitin Kharade
