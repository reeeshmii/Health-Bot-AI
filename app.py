import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from difflib import get_close_matches
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from streamlit.errors import StreamlitAPIException

# --- 1. CORE MACHINE LEARNING AND DATA LOADING (CACHED) ---

# Use Streamlit's caching to load the model and data only once
@st.cache_resource
def load_model_and_encoder():
    """Loads the trained RandomForest model and the LabelEncoder."""
    training = pd.read_csv('Training.csv')
    
    # Clean column names
    training.columns = training.columns.str.replace(r"[._\d]+$", "", regex=True).str.strip()
    training = training.loc[:, ~training.columns.duplicated()]

    cols = training.columns[:-1]
    y = training['prognosis']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    x = training[cols]
    x_train, _, y_train, _ = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x_train, y_train)
    
    return model, le, cols.tolist()

@st.cache_data
def load_helper_data():
    """Loads description, precaution dictionaries, and synonyms."""
    description_list = {}
    with open('symptom_Description.csv', encoding='utf-8') as csv_file:
        for row in csv_file:
            row = row.strip().split(',')
            if len(row) > 1:
                description_list[row[0]] = row[1]

    precautionDictionary = {}
    with open('symptom_precaution.csv', encoding='utf-8') as csv_file:
        for row in csv_file:
            row = row.strip().split(',')
            if len(row) > 1:
                precautionDictionary[row[0]] = row[1:]

    # Expanded synonyms for better matching
    symptom_synonyms = {
        "stomach ache": "stomach_pain", "belly pain": "stomach_pain", "tummy pain": "stomach_pain",
        "loose motion": "diarrhoea", "motions": "diarrhoea", "high temperature": "high_fever",
        "temperature": "mild_fever", "feaver": "mild_fever", "coughing": "cough",
        "throat pain": "sore_throat", "cold": "chills", "breathing issue": "breathlessness",
        "shortness of breath": "breathlessness", "body ache": "muscle_pain",
        "itchy": "itching", "skin bumps": "skin_rash", "bumpy skin": "skin_rash", "rash on skin": "skin_rash",
        "runny nose": "runny_nose", "head pain": "headache", "feeling tired": "fatigue",
        "tiredness": "fatigue", "throwing up": "vomiting", "feeling sick": "nausea", "puss ": "acne"
    }
    return description_list, precautionDictionary, symptom_synonyms

# --- 2. LLM INTEGRATION ---

# Configure Gemini API
try:
    # First, try to get the key from Streamlit's secrets
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except (StreamlitAPIException, KeyError):
    # If that fails, fall back to getting the key from environment variables
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("API Key not found. Please set the GEMINI_API_KEY environment variable or a Streamlit secret.")
        st.stop()

@st.cache_data
def get_llm_response(prompt: str) -> str:
    """Sends a prompt to the Gemini model and returns the response."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, an error occurred with the AI model: {e}"

# --- 3. HELPER FUNCTIONS ---

def extract_symptoms(user_input, all_symptoms, synonyms):
    """Extracts symptoms from user text using synonyms, direct, and fuzzy matching."""
    extracted = []
    text = user_input.lower().replace("_", " ").replace("-", " ")

    # Synonym matching
    for phrase, mapped in synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # Direct and fuzzy matching
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        if word in all_symptoms:
            extracted.append(word)
        else:
            close_matches = get_close_matches(word, all_symptoms, n=1, cutoff=0.8)
            if close_matches:
                extracted.append(close_matches[0])
    
    return list(set(extracted))

def predict_disease(symptoms_list, model, label_encoder, feature_names):
    """Creates an input vector and predicts the disease."""
    symptoms_dict = {symptom: idx for idx, symptom in enumerate(feature_names)}
    input_vector = np.zeros(len(feature_names))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    pred_proba = model.predict_proba(input_df)[0]
    pred_class_idx = np.argmax(pred_proba)
    disease = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = round(pred_proba[pred_class_idx] * 100, 2)
    return disease, confidence

# --- 4. STREAMLIT UI ---

# Load all data and models
model, le, all_symptoms = load_model_and_encoder()
descriptions, precautions, synonyms = load_helper_data()

# Initialize session state to manage the conversation flow
if 'stage' not in st.session_state:
    st.session_state.stage = 'initial'
    st.session_state.symptoms = []
    st.session_state.name = ""
    st.session_state.prediction = None
    st.session_state.asked_questions = []

st.set_page_config(page_title="Healio", page_icon="🩺")

# --- UI STAGES ---

# STAGE 0: Welcome and Initial Input
if st.session_state.stage == 'initial':
    st.title("🩺 Healio")
    st.info("Disclaimer: This AI is for informational purposes only and is not a substitute for professional medical advice. Always consult a doctor for health concerns.")
    
    st.session_state.name = st.text_input("What is your name?", key="user_name")
    user_input = st.text_area("Please describe your symptoms in a sentence or two:", height=100, key="symptom_input")
    
    if st.button("Analyze Symptoms", key="analyze_btn"):
        if not st.session_state.name:
            st.warning("Please enter your name.")
        elif not user_input:
            st.warning("Please describe your symptoms.")
        else:
            st.session_state.symptoms = extract_symptoms(user_input, all_symptoms, synonyms)
            if not st.session_state.symptoms:
                st.error("Sorry, I could not detect valid symptoms. Please try describing them differently.")
            else:
                st.session_state.stage = 'follow_up'
                st.rerun()

# STAGE 1: Follow-up Questions
elif st.session_state.stage == 'follow_up':
    st.title("Let's refine the analysis")
    st.write(f"Detected initial symptoms: **{', '.join(st.session_state.symptoms).replace('_', ' ')}**")
    
    disease, _ = predict_disease(st.session_state.symptoms, model, le, all_symptoms)
    
    # Get potential related symptoms
    training_df = pd.read_csv('Training.csv')
    training_df.columns = training_df.columns.str.replace(r"[._\d]+$", "", regex=True).str.strip()
    training_df = training_df.loc[:, ~training_df.columns.duplicated()]
    disease_row = training_df[training_df['prognosis'] == disease].iloc[0]
    potential_symptoms = [sym for sym in all_symptoms if disease_row[sym] == 1 and sym not in st.session_state.symptoms]
    
    st.session_state.asked_questions = potential_symptoms[:4] # Ask up to 4 relevant questions

    if not st.session_state.asked_questions:
        st.session_state.stage = 'result' # Skip if no more questions
        st.rerun()

    st.write("To help me narrow it down, please answer a few more questions:")
    
    with st.form("follow_up_form"):
        for i, sym in enumerate(st.session_state.asked_questions):
            st.session_state[f'q_{i}'] = st.radio(f"Do you also have **{sym.replace('_', ' ')}**?", ("No", "Yes"), key=f'q_{i}_radio', horizontal=True)
        
        submitted = st.form_submit_button("Get Final Prediction")
        if submitted:
            for i, sym in enumerate(st.session_state.asked_questions):
                if st.session_state[f'q_{i}'] == 'Yes':
                    st.session_state.symptoms.append(sym)
            st.session_state.stage = 'result'
            st.rerun()

# STAGE 2: Display Final Result
elif st.session_state.stage == 'result':
    st.title("Analysis Result")
    
    with st.spinner("Analyzing all symptoms and generating your personalized response..."):
        final_disease, final_confidence = predict_disease(st.session_state.symptoms, model, le, all_symptoms)
        
        # Construct the prompt for the LLM
        final_prompt = (
            f"You are an empathetic AI healthcare assistant. A user named {st.session_state.name} has received a "
            f"prediction from a model. Your task is to explain the result in a gentle, clear, and "
            f"reassuring way. Do not offer a diagnosis or medical advice. "
            f"Strictly adhere to the following structure:\n"
            f"1. Address the user by their name.\n"
            f"2. State what the analysis suggests (the disease name).\n"
            f"3. Briefly explain what the disease is using the provided description.\n"
            f"4. List the suggested precautions clearly using bullet points.\n"
            f"5. End with a strong, mandatory disclaimer that this is not a substitute for professional "
            f"medical advice and they should consult a doctor.\n\n"
            f"--- Information to use ---\n"
            f"Name: {st.session_state.name}\n"
            f"Predicted Condition: {final_disease}\n"
            f"Model Confidence: {final_confidence}%\n"
            f"Description: {descriptions.get(final_disease, 'No description available.')}\n"
            f"Precautions: {', '.join(precautions.get(final_disease, []))}\n"
        )
        
        final_response = get_llm_response(final_prompt)
        st.markdown(final_response)

    if st.button("Start Over", key="restart_btn"):
        st.session_state.clear()
        st.session_state.stage = 'initial'
        st.rerun()

