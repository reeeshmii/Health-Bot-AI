import streamlit as st
import re
import random
import pandas as pd
import numpy as np
import csv
import os
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings

# --- 1. SETUP AND CONFIGURATION ---

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
except (AttributeError, TypeError):
    st.error("GEMINI_API_KEY not found. Please set it as an environment variable or a Streamlit secret.")
    st.stop()

# --- 2. CACHED FUNCTIONS (FOR PERFORMANCE) ---

@st.cache_resource
def load_model_and_data():
    training = pd.read_csv('Training.csv')
    training.columns = training.columns.str.replace(r"[._\d]+$", "", regex=True).str.strip()
    training = training.loc[:, ~training.columns.duplicated()]

    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(x, y_encoded)
    
    return model, le, cols.tolist(), training
@st.cache_data
def load_helper_data():
    description_list = {}
    with open('symptom_Description.csv', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) > 1:
                description_list[row[0]] = row[1]
    
    precautionDictionary = {}
    with open('symptom_precaution.csv', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) > 1:
                precautionDictionary[row[0]] = row[1:]

    symptom_synonyms = {   # FIXED name and indentation
        # Original synonyms
        "stomach ache": "stomach_pain",
        "belly pain": "stomach_pain",
        "tummy pain": "stomach_pain",
        "loose motion": "diarrhoea",
        "motions": "diarrhoea",
        "high temperature": "high_fever",
        "temperature": "mild_fever",
        "feaver": "mild_fever",
        "coughing": "cough",
        "throat pain": "sore_throat",
        "cold": "chills",
        "breathing issue": "breathlessness",
        "shortness of breath": "breathlessness",
        "body ache": "muscle_pain",
        # --- NEW ADDITIONS TO FIX THE PROBLEM ---
        "itchy": "itching",
        "skin bumps": "skin_rash",
        "bumpy skin": "skin_rash",
        "rash on skin": "skin_rash",
        # --- OTHER USEFUL ADDITIONS ---
        "runny nose": "runny_nose",
        "head pain": "headache",
        "feeling tired": "fatigue",
        "tiredness": "fatigue",
        "throwing up": "vomiting",
        "feeling sick": "nausea",
        "bloating ": "Gastric issues"
    }
    return description_list, precautionDictionary, symptom_synonyms

# --- 3. CORE FUNCTIONS ---

def get_llm_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, an error occurred with the AI model: {e}"

def extract_symptoms(user_input, all_symptoms, synonyms):
    extracted = []
    text = user_input.lower().replace("_", " ").replace("-", " ")
    for phrase, mapped in synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        close_matches = get_close_matches(word, all_symptoms, n=1, cutoff=0.8)
        if close_matches:
            extracted.append(close_matches[0])
    return list(set(extracted))

def predict_disease(symptoms_list, model, label_encoder, feature_names):
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

model, le, cols, training_df = load_model_and_data()
description_list, precautionDictionary, symptom_synonyms = load_helper_data()

st.set_page_config(page_title="Health Insight Bot", page_icon="🩺")
st.title("🩺 Health Insight Bot")
st.info("Disclaimer: This AI is not a medical professional. Please consult a doctor for any health concerns.")

if 'stage' not in st.session_state:
    st.session_state.stage = 'initial'

if st.session_state.stage == 'initial':
    name = st.text_input("What is your name?")
    symptoms_input_str = st.text_area("Describe your symptoms:", height=100)
    
    if st.button("Analyze Symptoms"):
        if name and symptoms_input_str:
            st.session_state.name = name
            st.session_state.symptoms = extract_symptoms(symptoms_input_str, cols, symptom_synonyms)
            
            if st.session_state.symptoms:
                st.session_state.stage = 'follow_up'
                st.rerun()
            else:
                st.error("Could not detect valid symptoms. Please describe them differently.")
        else:
            st.warning("Please enter your name and describe your symptoms.")

if st.session_state.stage == 'follow_up':
    st.write(f"Hello {st.session_state.name}!")
    st.info(f"Detected Symptoms: **{', '.join(st.session_state.symptoms)}**")
    
    disease, _ = predict_disease(st.session_state.symptoms, model, le, cols)
    disease_row = training_df[training_df['prognosis'] == disease].iloc[0]
    potential_symptoms = [sym for sym in cols if disease_row[sym] == 1 and sym not in st.session_state.symptoms]
    
    st.markdown("---")
    st.write("To refine the analysis, please answer a few more questions:")
    
    with st.form("follow_up_form"):
        for sym in potential_symptoms[:4]:
            st.radio(
                f"Do you also have **{sym.replace('_', ' ')}**?",
                ("Yes", "No"),
                key=f'q_{sym}',
                index=None,  # This is the fix! It makes the default unselected.
                horizontal=True
            )
        
        submitted = st.form_submit_button("Get Final Result")
        if submitted:
            for sym in potential_symptoms[:4]:
                if st.session_state.get(f'q_{sym}') == 'Yes':
                    st.session_state.symptoms.append(sym)
            st.session_state.stage = 'result'
            st.rerun()

if st.session_state.stage == 'result':
    st.title("Analysis Result")
    
    with st.spinner("Analyzing and generating your personalized response..."):
        disease, confidence = predict_disease(st.session_state.symptoms, model, le, cols)
        precautions = precautionDictionary.get(disease, [])
        description = description_list.get(disease, "No description available.")
        
        final_prompt = (
            f"You are an empathetic AI healthcare assistant. A user named {st.session_state.name} has received a prediction. "
            f"Explain the result gently and clearly. List the precautions using bullet points. "
            f"End with a strong disclaimer to consult a doctor. Do not offer a diagnosis.\n\n"
            f"--- Information to use ---\n"
            f"Name: {st.session_state.name}\n"
            f"Predicted Condition: {disease}\n"
            f"Model Confidence: {confidence}%\n"
            f"Description: {description}\n"
            f"Precautions: {', '.join(precautions)}\n"
        )
        
        final_response = get_llm_response(final_prompt)
        st.markdown(final_response)

    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()

