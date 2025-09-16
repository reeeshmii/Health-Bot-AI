🩺 Healio
An intelligent, conversational health chatbot that leverages machine learning and a Large Language Model (LLM) to provide preliminary insights based on user-described symptoms. This project features a user-friendly web interface built with Streamlit.

(Suggestion: Replace the placeholder above with a real screenshot of your running Streamlit application!)

✨ Key Features
Natural Language Symptom Analysis: Users can describe their symptoms in plain English. The bot uses a custom NLP function to extract and standardize relevant medical symptoms.

Machine Learning Predictions: A RandomForestClassifier model, trained on a comprehensive dataset of over 4,900 symptom-prognosis entries, predicts potential conditions.

Dynamic & Empathetic Responses: Integrates with the Google Gemini API to generate conversational, context-aware, and reassuring responses, transforming raw predictions into clear, helpful information.

Interactive Web Interface: A clean and intuitive UI built with Streamlit guides the user through a multi-step process, from initial input to the final, detailed analysis.

Built-in Safeguards: The application includes a prominent disclaimer and is designed to always direct users to consult with a professional doctor.

🛠️ Technologies Used
Backend: Python

Machine Learning: Scikit-learn, Pandas, NumPy

Web Framework: Streamlit

LLM Integration: Google Gemini API

Data Files: CSV

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

1. Prerequisites
Python 3.8 or higher

A Google Gemini API Key. You can get one from Google AI Studio.

2. API Key Setup (Crucial)
This application requires a Google Gemini API key to function. The most secure way to handle this is using Streamlit's built-in secrets management.

Create a new folder in your project's root directory named .streamlit.

Inside the .streamlit folder, create a new file named secrets.toml.

Open secrets.toml and add your API key in the following format:

GEMINI_API_KEY = "YOUR_API_KEY_HERE"

The application will automatically load this key.

3. Running the Application
Once the setup is complete, run the following command in your terminal from the project's root directory:

streamlit run app.py

Your web browser will automatically open with the Health Insight Bot interface.

📜 Disclaimer
This project is an educational tool and a demonstration of machine learning and API integration capabilities. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or another qualified health provider with any questions you may have regarding a medical condition.
