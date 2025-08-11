import numpy as np
import pandas as pd
import statistics
import warnings
import sqlite3
import difflib
from collections import Counter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess dataset
data = pd.read_csv("data/Disease_Data.csv").dropna(axis=1)

encoder = LabelEncoder()
data['prognosis'] = encoder.fit_transform(data['prognosis'])

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train models
final_SVM_model = SVC(probability=True)
final_NB_model = GaussianNB()
final_RF_model = RandomForestClassifier(random_state=18)

final_SVM_model.fit(x_train, y_train)
final_NB_model.fit(x_train, y_train)
final_RF_model.fit(x_train, y_train)

# Mapping
symptom_index = {symptom.strip().lower().replace(" ", "_"): idx for idx, symptom in enumerate(x.columns)}
predictions_classes = {index: disease for index, disease in enumerate(encoder.classes_)}

# Dummy data for example
symptom_aliases = {
    'itching': ['scratching', 'skin itch', 'pruritus'],
    'skin_rash': ['rash', 'skin irritation', 'red skin', 'skin eruption'],
    'nodal_skin_eruptions': ['skin nodules', 'bumps on skin', 'skin lumps'],
    'continuous_sneezing': ['sneezing', 'frequent sneezing', 'sneeze'],

}

symptom_clusters = {
    1: ["fever", "cough", "fatigue"],
    2: ["nausea", "vomiting", "diarrhea"],
}
disease_info = {
    "Fungal Infection": {
        "description": "Fungal infections can cause a variety of symptoms. Treatments usually include antifungal medications.",
        "lab_tests": ["KOH Test", "Fungal Culture", "PAS Staining"],
        "telemedicine_keywords": ["dermatology", "infectious disease"]
    },
   
    
    "Common Cold": {
        "description": "Management includes rest and hydration; no specific cure exists.",
        "lab_tests": ["Usually none required", "Throat Swab if bacterial suspected"],
        "telemedicine_keywords": ["family medicine", "internal medicine"]
    },
  
}

# Admin and chatbot
admin_user = {"username": "admin", "email": "admin@example.com", "password": "admin123"}
responses = {
    "greeting": ["hello", "hi", "hey", "greetings"],
    "wellness": ["how are you?", "how do you do?", "what's up?"],
    "capabilities": ["what can you do?", "tell me your capabilities"],
    "symptoms": ["symptoms", "i have symptoms", "i feel sick", "i am unwell"],
    "health_advice": ["health advice", "tips", "recommendations"],
    "goodbye": ["bye", "goodbye", "see you", "take care"],
    "scheduling": ["schedule", "i need help with my schedule", "can you help me plan"],
    "diet": ["diet", "nutrition", "meal plan", "what should I eat"],
}

def normalize_symptom(symptom):
    symptom = symptom.strip().lower().replace(" ", "_")
    if symptom in symptom_index:
        return symptom
    for standard_symptom, aliases in symptom_aliases.items():
        if symptom in [alias.replace(" ", "_") for alias in aliases]:
            return standard_symptom
    possible_symptoms = list(symptom_index.keys()) + list(symptom_aliases.keys())
    matches = difflib.get_close_matches(symptom, possible_symptoms, n=1, cutoff=0.7)
    return matches[0] if matches else None

def get_related_symptoms(current_symptoms):
    related = []
    current_clusters = set()
    for symptom in current_symptoms:
        for cluster_id, symptoms in symptom_clusters.items():
            if symptom in symptoms:
                current_clusters.add(cluster_id)
    for cluster_id in current_clusters:
        for symptom in symptom_clusters[cluster_id]:
            if symptom not in current_symptoms:
                related.append(symptom)
    return related[:10]

def calculate_confidence_interval(probabilities, confidence_level=0.95):
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    n = len(probabilities)
    alpha = 1 - confidence_level
    t_score = stats.t.ppf(1 - alpha/2, n-1)
    margin_error = t_score * (std_prob / np.sqrt(n))
    ci_lower = max(0, mean_prob - margin_error)
    ci_upper = min(1, mean_prob + margin_error)
    return ci_lower, ci_upper, mean_prob

def generate_intelligent_questions(symptoms, age, gender, previous_responses=None):
    related_symptoms = get_related_symptoms(symptoms)
    # Dummy question generation
    return [
        "How long have you been experiencing these symptoms?",
        "On a scale of 1-10, how severe is your discomfort?",
        "Do you have any family history of similar conditions?",
        "Are you currently taking any medications?",
        "Have you noticed any triggers that make symptoms worse?"
    ]

def enhanced_predictDisease(symptoms, age=None, gender=None):
    if not final_RF_model:
        return {"error": "Models not loaded. Please check if the dataset is available."}
    
    symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    normalized_symptoms, unrecognized_symptoms = [], []

    for symptom in symptoms_list:
        normalized = normalize_symptom(symptom)
        if normalized and normalized in symptom_index:
            normalized_symptoms.append(normalized)
        else:
            unrecognized_symptoms.append(symptom)

    if not normalized_symptoms:
        return {
            "error": "No recognized symptoms found",
            "unrecognized_symptoms": unrecognized_symptoms,
            "suggestion": "Try describing symptoms differently or check spelling",
            "available_symptoms": list(symptom_index.keys())[:20]
        }

    input_data = [0] * len(symptom_index)
    matched_symptoms = []

    for symptom in normalized_symptoms:
        if symptom in symptom_index:
            index = symptom_index[symptom]
            input_data[index] = 1
            matched_symptoms.append(symptom)

    input_data = np.array(input_data).reshape(1, -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_prediction = predictions_classes[final_RF_model.predict(input_data)[0]]
        nb_prediction = predictions_classes[final_NB_model.predict(input_data)[0]]
        svm_prediction = predictions_classes[final_SVM_model.predict(input_data)[0]]

        rf_proba = final_RF_model.predict_proba(input_data)[0]
        nb_proba = final_NB_model.predict_proba(input_data)[0]
        svm_proba = final_SVM_model.predict_proba(input_data)[0]

        all_probas = np.array([rf_proba, nb_proba, svm_proba])
        ensemble_proba = np.mean(all_probas, axis=0)
        ensemble_prediction = predictions_classes[np.argmax(ensemble_proba)]

        max_probas = [np.max(prob) for prob in all_probas]
        ci_lower, ci_upper, mean_confidence = calculate_confidence_interval(max_probas)

    predictions = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = Counter(predictions).most_common(1)[0][0]
    related_symptoms = get_related_symptoms(matched_symptoms)
    questions = generate_intelligent_questions(matched_symptoms, age, gender) if age and gender else []

    disease_details = disease_info.get(final_prediction, {
        "description": "No info available. Consult a professional.",
        "lab_tests": ["Consult doctor"],
        "telemedicine_keywords": ["general"]
    })

    return {
        "prediction_type": "ensemble_ml_model",
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "ensemble_prediction": ensemble_prediction,
        "final_prediction": final_prediction,
        "confidence": f"{mean_confidence:.3f}",
        "confidence_interval": f"[{ci_lower:.3f}, {ci_upper:.3f}]",
        "disease_info": disease_details["description"],
        "lab_tests": disease_details["lab_tests"],
        "telemedicine_keywords": disease_details["telemedicine_keywords"],
        "matched_symptoms": matched_symptoms,
        "unrecognized_symptoms": unrecognized_symptoms,
        "related_symptoms": related_symptoms,
        "intelligent_questions": questions,
        "recommendation": "Consult a healthcare professional for accurate diagnosis."
    }

def generate_response(user_input):
    for category, keywords in responses.items():
        for keyword in keywords:
            if keyword in user_input.lower():
                return generate_response_for_category(category)
    return "I'm not sure how to help with that. Can you please rephrase?"

def generate_response_for_category(category):
    if category == "greeting":
        return "Hello! I'm Vitalo, your health assistant. How can I help today?"
    elif category == "wellness":
        return "I'm here to support you 24/7!"
    elif category == "capabilities":
        return "I can provide symptom checks, health advice, and connect you with professionals."
    elif category == "symptoms":
        return "Please describe your symptoms."
    elif category == "health_advice":
        return "Stay hydrated, rest well, and eat a balanced diet."
    elif category == "goodbye":
        return "Goodbye! Take care!"
    elif category == "scheduling":
        return "Tell me your needs, I can help plan your schedule."
    elif category == "diet":
        return "Tell me your food habits, I’ll give dietary advice!"
    return "I'm not sure how to help with that."

def init_db():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            location TEXT,
            suggestion TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()