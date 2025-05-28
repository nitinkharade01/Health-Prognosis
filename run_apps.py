import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def run_flask():
    """Run the Flask app"""
    flask_process = subprocess.Popen([sys.executable, 'app.py'])
    return flask_process

def run_streamlit():
    """Run the Streamlit app"""
    streamlit_process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
    return streamlit_process

def open_browsers():
    """Open browser windows for both apps"""
    time.sleep(5)  # Wait for servers to start
    webbrowser.open('http://localhost:5000')  # Flask app
    webbrowser.open('http://localhost:8501')  # Streamlit app

if __name__ == '__main__':
    print("Starting Flask and Streamlit apps...")
    
    # Start Flask app
    flask_process = run_flask()
    print("Flask app started at http://localhost:5000")
    
    # Start Streamlit app
    streamlit_process = run_streamlit()
    print("Streamlit app started at http://localhost:8501")
    
    # Open browsers in a separate thread
    Thread(target=open_browsers).start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        flask_process.terminate()
        streamlit_process.terminate()
        print("Servers stopped.") 