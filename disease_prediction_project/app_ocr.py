import numpy as np
import pandas as pd
import warnings
import sqlite3
import difflib
import random
import statistics
from collections import Counter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session


import google.generativeai as genai
# Replace with your actual Gemini API key
genai.configure(api_key="api key")
model = genai.GenerativeModel("gemini-2.5-flash")

chat = model.start_chat(history=[])
system_prompt = """
You are Vitalo, a smart and friendly virtual health assistant.
Your job is to respond conversationally and compassionately to users' questions or symptoms.
You should:
•⁠  ⁠Give wellness guidance
•⁠  ⁠Help users describe symptoms
•⁠  ⁠Recommend seeing doctors (but never diagnose)
•⁠  ⁠Talk like a caring, intelligent assistant
"""
chat.send_message(system_prompt)
# === Load and preprocess dataset function ===
def load_and_preprocess_data():
    try:
        data = pd.read_csv("/Users/karankalawant/Desktop/disease_prediction_project/data/Disease_Data.csv").dropna(axis=1)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        return X_train, X_test, y_train, y_test, encoder, X.columns.tolist()

    except FileNotFoundError:
        print("Error: Disease_Data.csv file not found.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

# === Load data ===
X_train, X_test, y_train, y_test, encoder, symptom_columns = load_and_preprocess_data()

# === Initialize and train models if data loaded ===
final_SVM_model = final_NB_model = final_RF_model = None
symptom_index = {}
predictions_classes = {}

if X_train is not None:
    base_svm = SVC(probability=True, class_weight='balanced', random_state=42)
    final_SVM_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    final_NB_model = GaussianNB()
    final_RF_model = OneVsRestClassifier(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    )

    final_SVM_model.fit(X_train, y_train)
    final_NB_model.fit(X_train, y_train)
    final_RF_model.fit(X_train, y_train)

    symptom_index = {symptom.strip().lower().replace(" ", "_"): idx for idx, symptom in enumerate(symptom_columns)}
    predictions_classes = {index: disease for index, disease in enumerate(encoder.classes_)}

symptom_clusters = {
    "Skin & Allergies": [
        "itching",
        "skin_rash",
        "nodal_skin_eruptions",
        "dry_skin",
        "scaly_skin",
        "pus_filled_pimples",
        "small_dents_in_nails",
        "inflammatory_nails",
        "blister",
        "skin_peeling",
        "silver_like_dusting"
    ],
    
    "Respiratory": [
        "continuous_sneezing",
        "cough",
        "breathlessness",
        "phlegm",
        "chest_pain",
        "wheezing",
        "throat_irritation",
        "sinus_pressure"
    ],
    
    "Gastrointestinal": [
        "stomach_pain",
        "vomiting",
        "nausea",
        "acidity",
        "indigestion",
        "diarrhoea",
        "constipation",
        "abdominal_pain",
        "ulcers_on_tongue",
        "belly_pain",
        "bloody_stool"
    ],
    
    "Neurological": [
        "headache",
        "dizziness",
        "loss_of_balance",
        "unsteadiness",
        "numbness",
        "slurred_speech",
        "mood_swings",
        "anxiety",
        "depression",
        "lack_of_concentration",
        "restlessness",
        "irritability",
        "altered_sensorium"
    ],
    
    "Musculoskeletal": [
        "joint_pain",
        "back_pain",
        "muscle_pain",
        "muscle_wasting",
        "swollen_legs",
        "stiff_neck",
        "weakness_in_limbs",
        "knee_pain",
        "hip_joint_pain"
    ],
    
    "Urological & Renal": [
        "burning_micturition",
        "spotting_urination",
        "dark_urine",
        "foul_smell_of_urine",
        "bladder_discomfort",
        "pain_in_anal_region"
    ],
    
    "Cardiovascular": [
        "chest_pain",
        "palpitations",
        "high_blood_pressure",
        "low_blood_pressure",
        "sweating",
        "fatigue"
    ],
    
    "Endocrine & Metabolic": [
        "weight_gain",
        "weight_loss",
        "lethargy",
        "cold_hands_and_feets",
        "irregular_sugar_level",
        "excessive_hunger",
        "increased_appetite"
    ],
    
    "Eye & ENT": [
        "sunken_eyes",
        "blurred_and_distorted_vision",
        "watering_from_eyes",
        "redness_of_eyes",
        "patches_in_throat",
        "swollen_lymph_nodes",
        "runny_nose"
    ],
    
    "General/Systemic": [
        "fever",
        "high_fever",
        "chills",
        "shivering",
        "fatigue",
        "malaise",
        "dehydration",
        "yellowing_of_eyes",
        "yellowish_skin",
        "toxicity"
    ],
    
    "Infections": [
        "pus_discharge",
        "dischromic_patches",
        "skin_infection",
        "nodal_skin_eruptions",
        "fungal_infection",
        "enlarged_lymph_nodes",
        "inflammatory_swelling"
    ]
}

disease_info = {
    "Fungal Infection": {
        "description": "Fungal infections can cause a variety of symptoms. Treatments usually include antifungal medications.",
        "lab_tests": ["KOH Test", "Fungal Culture", "PAS Staining"],
        "telemedicine_keywords": ["dermatology", "infectious disease"]
    },
    "Allergy": {
        "description": "Allergies can manifest as sneezing, itching, and swelling. Antihistamines are often effective.",
        "lab_tests": ["IgE Test", "Skin Prick Test", "RAST Test"],
        "telemedicine_keywords": ["allergy", "immunology"]
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease can cause heartburn and regurgitation. Lifestyle changes and medications can help.",
        "lab_tests": ["Upper Endoscopy", "pH Monitoring", "Barium Swallow"],
        "telemedicine_keywords": ["gastroenterology", "internal medicine"]
    },
    "Chronic Cholestasis": {
        "description": "This condition involves bile flow blockage and may require dietary changes and medications.",
        "lab_tests": ["Liver Function Tests", "ALP", "GGT", "Bilirubin"],
        "telemedicine_keywords": ["hepatology", "gastroenterology"]
    },
    "Drug Reaction": {
        "description": "Adverse drug reactions vary widely; consult a doctor for management and potential alternatives.",
        "lab_tests": ["Drug Level Monitoring", "Liver Function Tests", "CBC"],
        "telemedicine_keywords": ["clinical pharmacology", "allergy"]
    },
    "Peptic Ulcer Disease": {
        "description": "This involves sores on the stomach lining, typically treated with antacids or antibiotics.",
        "lab_tests": ["H. Pylori Test", "Upper Endoscopy", "Stool Antigen Test"],
        "telemedicine_keywords": ["gastroenterology", "internal medicine"]
    },
    "AIDS": {
        "description": "Acquired Immunodeficiency Syndrome requires lifelong antiretroviral therapy.",
        "lab_tests": ["HIV Test", "CD4 Count", "Viral Load", "Complete Blood Count"],
        "telemedicine_keywords": ["infectious disease", "HIV specialist"]
    },
    "Diabetes": {
        "description": "Diabetes management involves lifestyle changes and medications to regulate blood sugar.",
        "lab_tests": ["HbA1c", "Fasting Glucose", "OGTT", "Lipid Profile"],
        "telemedicine_keywords": ["endocrinology", "diabetes care"]
    },
    "Gastroenteritis": {
        "description": "This causes stomach flu symptoms and is often managed with hydration and dietary adjustments.",
        "lab_tests": ["Stool Culture", "Stool Ova & Parasites", "Electrolytes"],
        "telemedicine_keywords": ["gastroenterology", "internal medicine"]
    },
    "Bronchial Asthma": {
        "description": "Asthma involves airway inflammation; inhalers and avoiding triggers can help.",
        "lab_tests": ["Spirometry", "Peak Flow", "IgE Test", "Chest X-ray"],
        "telemedicine_keywords": ["pulmonology", "allergy"]
    },
    "Hypertension": {
        "description": "High blood pressure may require lifestyle changes and medications.",
        "lab_tests": ["ECG", "Echocardiogram", "Lipid Profile", "Kidney Function Tests"],
        "telemedicine_keywords": ["cardiology", "internal medicine"]
    },
    "Migraine": {
        "description": "Migraine treatments include pain relief and preventive medications.",
        "lab_tests": ["MRI Brain", "CT Scan", "Blood Tests for triggers"],
        "telemedicine_keywords": ["neurology", "headache specialist"]
    },
    "Cervical Spondylosis": {
        "description": "This condition affects the neck vertebrae; physical therapy can be beneficial.",
        "lab_tests": ["X-ray Cervical Spine", "MRI Cervical Spine", "CT Scan"],
        "telemedicine_keywords": ["orthopedics", "neurology"]
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Rehabilitation and therapy are essential for recovery.",
        "lab_tests": ["CT Brain", "MRI Brain", "Coagulation Studies", "ECG"],
        "telemedicine_keywords": ["neurology", "neurosurgery"]
    },
    "Jaundice": {
        "description": "This indicates liver issues; treatment depends on the underlying cause.",
        "lab_tests": ["Liver Function Tests", "Bilirubin", "Hepatitis Panel", "CBC"],
        "telemedicine_keywords": ["hepatology", "gastroenterology"]
    },
    "Malaria": {
        "description": "Preventable with mosquito nets; treated with antimalarial medications.",
        "lab_tests": ["Malaria Parasite Test", "Rapid Diagnostic Test", "Blood Smear"],
        "telemedicine_keywords": ["infectious disease", "tropical medicine"]
    },
    "Chicken Pox": {
        "description": "Chickenpox usually resolves on its own but can be prevented with vaccines.",
        "lab_tests": ["Varicella-Zoster Virus PCR", "Tzanck Smear"],
        "telemedicine_keywords": ["dermatology", "infectious disease"]
    },
    "Dengue": {
        "description": "Dengue fever treatment involves managing symptoms; prevention focuses on mosquito control.",
        "lab_tests": ["Dengue NS1 Antigen", "IgM/IgG Antibodies", "Platelet Count"],
        "telemedicine_keywords": ["infectious disease", "tropical medicine"]
    },
    "Typhoid": {
        "description": "Typhoid fever requires antibiotics and can be prevented with vaccines.",
        "lab_tests": ["Widal Test", "Blood Culture", "Stool Culture", "Typhidot"],
        "telemedicine_keywords": ["infectious disease", "internal medicine"]
    },
    "Hepatitis A": {
        "description": "Vaccination and hygiene are key in prevention; treatment focuses on managing symptoms.",
        "lab_tests": ["Hepatitis A IgM/IgG", "Liver Function Tests"],
        "telemedicine_keywords": ["hepatology", "infectious disease"]
    },
    "Hepatitis B": {
        "description": "Vaccination is effective; treatment may involve antiviral medications.",
        "lab_tests": ["HBsAg", "HBeAg", "Anti-HBc", "HBV DNA"],
        "telemedicine_keywords": ["hepatology", "infectious disease"]
    },
    "Hepatitis C": {
        "description": "Direct-acting antiviral treatments are effective for most cases.",
        "lab_tests": ["Anti-HCV", "HCV RNA", "Genotyping"],
        "telemedicine_keywords": ["hepatology", "infectious disease"]
    },
    "Hepatitis D": {
        "description": "This requires managing hepatitis B, as it's dependent on it.",
        "lab_tests": ["Anti-HDV", "HDV RNA", "HBsAg"],
        "telemedicine_keywords": ["hepatology", "infectious disease"]
    },
    "Hepatitis E": {
        "description": "Prevention focuses on clean water; treatment is typically supportive.",
        "lab_tests": ["Anti-HEV IgM/IgG", "HEV RNA"],
        "telemedicine_keywords": ["hepatology", "infectious disease"]
    },
    "Alcoholic Hepatitis": {
        "description": "Treatment involves abstaining from alcohol and managing symptoms.",
        "lab_tests": ["Liver Function Tests", "PT/INR", "Albumin", "AST/ALT Ratio"],
        "telemedicine_keywords": ["hepatology", "addiction medicine"]
    },
    "Tuberculosis": {
        "description": "Requires long-term antibiotics; prevention includes vaccination.",
        "lab_tests": ["Sputum AFB", "Chest X-ray", "Mantoux Test", "GeneXpert"],
        "telemedicine_keywords": ["pulmonology", "infectious disease"]
    },
    "Common Cold": {
        "description": "Management includes rest and hydration; no specific cure exists.",
        "lab_tests": ["Usually none required", "Throat Swab if bacterial suspected"],
        "telemedicine_keywords": ["family medicine", "internal medicine"]
    },
    "Pneumonia": {
        "description": "Antibiotics may be required for bacterial pneumonia; prevention includes vaccines.",
        "lab_tests": ["Chest X-ray", "CBC", "Blood Culture", "Sputum Culture"],
        "telemedicine_keywords": ["pulmonology", "internal medicine"]
    },
    "Hemorrhoids (Piles)": {
        "description": "Treatment includes dietary changes and topical medications.",
        "lab_tests": ["Digital Rectal Exam", "Colonoscopy", "CBC"],
        "telemedicine_keywords": ["gastroenterology", "colorectal surgery"]
    },
    "Heart Attack": {
        "description": "Immediate medical attention is crucial; lifestyle changes can prevent further attacks.",
        "lab_tests": ["ECG", "Troponin", "CK-MB", "Echocardiogram"],
        "telemedicine_keywords": ["cardiology", "emergency medicine"]
    },
    "Varicose Veins": {
        "description": "Compression stockings and lifestyle changes may alleviate symptoms.",
        "lab_tests": ["Doppler Ultrasound", "Venography"],
        "telemedicine_keywords": ["vascular surgery", "dermatology"]
    },
    "Hypothyroidism": {
        "description": "Thyroid hormone replacement is standard treatment.",
        "lab_tests": ["TSH", "Free T4", "Free T3", "Anti-TPO"],
        "telemedicine_keywords": ["endocrinology", "internal medicine"]
    },
    "Hyperthyroidism": {
        "description": "Management includes medications, radioactive iodine, or surgery.",
        "lab_tests": ["TSH", "Free T4", "Free T3", "Thyroid Scan"],
        "telemedicine_keywords": ["endocrinology", "internal medicine"]
    },
    "Hypoglycemia": {
        "description": "Immediate intake of sugar is essential; long-term management involves dietary changes.",
        "lab_tests": ["Random Glucose", "HbA1c", "C-peptide", "Insulin Level"],
        "telemedicine_keywords": ["endocrinology", "internal medicine"]
    },
    "Osteoarthritis": {
        "description": "Joint pain management includes physical therapy and medications.",
        "lab_tests": ["X-ray Joints", "MRI", "Joint Fluid Analysis"],
        "telemedicine_keywords": ["rheumatology", "orthopedics"]
    },
    "Arthritis": {
        "description": "Treatment focuses on pain relief and improving joint function.",
        "lab_tests": ["ESR", "CRP", "RF", "Anti-CCP", "X-ray"],
        "telemedicine_keywords": ["rheumatology", "internal medicine"]
    },
    "(Vertigo) Paroxysmal Positional Vertigo": {
        "description": "Vestibular rehabilitation and maneuvers can help manage symptoms.",
        "lab_tests": ["Audiometry", "VNG", "MRI Brain if needed"],
        "telemedicine_keywords": ["ENT", "neurology"]
    },
    "Acne": {
        "description": "Topical treatments and medications are effective for management.",
        "lab_tests": ["Usually clinical diagnosis", "Hormone levels if severe"],
        "telemedicine_keywords": ["dermatology", "cosmetic dermatology"]
    },
    "Urinary Tract Infection": {
        "description": "Antibiotics are typically required; increased hydration helps.",
        "lab_tests": ["Urine Culture", "Urinalysis", "CBC"],
        "telemedicine_keywords": ["urology", "internal medicine"]
    },
    "Psoriasis": {
        "description": "Topical treatments and phototherapy can help manage this chronic condition.",
        "lab_tests": ["Skin Biopsy", "CBC", "Liver Function Tests"],
        "telemedicine_keywords": ["dermatology", "rheumatology"]
    },
    "Impetigo": {
        "description": "Antibiotics are necessary for treatment of this skin infection.",
        "lab_tests": ["Bacterial Culture", "Gram Stain"],
        "telemedicine_keywords": ["dermatology", "infectious disease"]
    }
}

# Enhanced symptom mapping (keeping your existing one)
symptom_aliases = {
    'itching': ['scratching', 'skin itch', 'pruritus'],
    'skin_rash': ['rash', 'skin irritation', 'red skin', 'skin eruption'],
    'nodal_skin_eruptions': ['skin nodules', 'bumps on skin', 'skin lumps'],
    'continuous_sneezing': ['sneezing', 'frequent sneezing', 'sneeze'],
    'shivering': ['chills', 'trembling', 'cold shivers'],
    'chills': ['cold feeling', 'feeling cold', 'shivering'],
    'joint_pain': ['arthritis pain', 'joint ache', 'bone pain'],
    'stomach_pain': ['abdominal pain', 'belly pain', 'stomach ache', 'tummy pain'],
    'acidity': ['acid reflux', 'heartburn', 'sour stomach'],
    'ulcers_on_tongue': ['tongue sores', 'mouth ulcers', 'oral ulcers'],
    'muscle_wasting': ['muscle loss', 'muscle atrophy', 'muscle weakness'],
    'vomiting': ['throwing up', 'puking', 'emesis', 'nausea vomiting'],
    'burning_micturition': ['painful urination', 'burning pee', 'dysuria'],
    'spotting_urination': ['blood in urine', 'hematuria', 'bloody urine'],
    'fatigue': ['tired', 'exhausted', 'weakness', 'weary', 'drained'],
    'weight_gain': ['gaining weight', 'getting heavier'],
    'anxiety': ['worry', 'nervousness', 'panic', 'stress'],
    'cold_hands_and_feets': ['cold extremities', 'cold hands feet'],
    'mood_swings': ['emotional changes', 'mood changes'],
    'weight_loss': ['losing weight', 'getting thinner'],
    'restlessness': ['agitation', 'inability to rest'],
    'lethargy': ['sluggishness', 'lack of energy'],
    'patches_in_throat': ['throat patches', 'throat spots'],
    'irregular_sugar_level': ['blood sugar fluctuation', 'glucose imbalance'],
    'cough': ['coughing', 'hack', 'persistent cough'],
    'high_fever': ['fever', 'high temperature', 'pyrexia'],
    'sunken_eyes': ['hollow eyes', 'deep set eyes'],
    'breathlessness': ['shortness of breath', 'difficulty breathing', 'dyspnea'],
    'sweating': ['perspiration', 'excessive sweating'],
    'dehydration': ['fluid loss', 'lack of fluids'],
    'indigestion': ['stomach upset', 'digestive problems'],
    'headache': ['head pain', 'migraine', 'head ache', 'cranial pain']
}


admin_user = {"username": "admin", "email": "admin@example.com", "password": "admin123"}

responses = {
    "greeting": [
        "hello", "hi", "hey", "greetings", "morning", "evening", "afternoon", "yo", "hiya", "sup"
    ],

    "wellness": [
        "wellness", "feeling", "howdy", "vibe", "energy", "status", "mood", "okay", "fine", "tired", "unwell"
    ],

    "capabilities": [
        "help", "features", "can do", "services", "assist", "capabilities", "support", "tools", "functions"
    ],

    "symptoms": [
        "symptoms", "pain", "fever", "headache", "stomachache", "chills", "cough", "cold", "dizzy", 
        "tired", "weak", "vomit", "nausea", "fatigue", "sore", "rash", "itch", "throat", "breathing"
    ],

    "health_advice": [
        "advice", "tips", "recommendations", "fitness", "healthy", "wellness", "preventive", 
        "lifestyle", "hydration", "exercise", "routine", "rest", "care", "prevention"
    ],

    "goodbye": [
        "bye", "goodbye", "see ya", "later", "farewell", "take care", "ciao", "adios", "peace", "exit"
    ],

    "scheduling": [
        "schedule", "appointment", "booking", "calendar", "reminder", "checkup", "visit", "plan", 
        "slot", "timing", "reschedule", "followup"
    ],

    "diet": [
        "diet", "nutrition", "food", "meal", "calories", "protein", "fiber", "vegetables", "fruits", 
        "balanced", "weight", "diabetes", "plan", "eating", "healthy", "carbs", "keto", "vegan"
    ],

    "mental_health": [
        "mental", "stress", "anxiety", "depression", "sad", "sleep", "insomnia", "panic", 
        "overwhelmed", "therapy", "support", "mood", "burnout", "tension"
    ],

    "medications": [
        "medicine", "medication", "tablet", "pill", "dose", "side effects", "drug", "prescription", 
        "pharmacy", "antibiotic", "painkiller", "capsule", "injection", "syrup"
    ],

    "emergency": [
        "emergency", "ambulance", "urgent", "pain", "stroke", "bleeding", "heart", "attack", "fainted", 
        "breathing", "choking", "collapse", "danger", "trauma"
    ]
}

def generate_response(user_input):
    try:
        response = chat.send_message(user_input)
        return response.text.strip()
    except Exception as e:
        return f"Oops, something went wrong. (Error: {str(e)})"
        

# === Helper Functions ===
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
    return max(0, mean_prob - margin_error), min(1, mean_prob + margin_error), mean_prob

def generate_intelligent_questions(symptoms, age, gender, previous_responses=None):
    return [
        "How long have you been experiencing these symptoms?",
        "On a scale of 1-10, how severe is your discomfort?",
        "Do you have any family history of similar conditions?",
        "Are you currently taking any medications?",
        "Have you noticed any triggers that make symptoms worse?"
    ]

def predictDisease(symptoms, age=None, gender=None):
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

        ensemble_proba = np.mean([rf_proba, nb_proba, svm_proba], axis=0)
        ensemble_prediction = predictions_classes[np.argmax(ensemble_proba)]
        max_probas = [np.max(p) for p in [rf_proba, nb_proba, svm_proba]]
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

# === Flask Web App ===
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Initialize database
init_db()

# Import required modules for prescription scanning
import os
import base64
import requests
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load .env variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def make_gemini_request(image_base64):
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "text": (
                            "From this prescription image, extract all prescribed medicine names, "
                            "their dosage frequency (e.g., 1-0-1), and a short one-line use case for each. "
                            "Return the output like: tablet name: , frequency, use case for each tablet on a new line each."
                        )
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    return response

# === Flask Routes ===
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO Users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
            conn.commit()
            flash("Signup successful! Please log in.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.")
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Users WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session["username"] = user[1]
            flash("Logged in successfully!")
            return redirect(url_for("home"))
        elif email == admin_user["email"] and password == admin_user["password"]:
            session["username"] = admin_user["username"]
            flash("Admin logged in!")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out.")
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        flash("Login required.")
        return redirect(url_for("login"))
    result = {}
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        result = predictDisease(symptoms)
    return render_template("index.html", result=result)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if "username" not in session:
        flash("Login to send feedback.")
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        location = request.form["location"]
        suggestion = request.form["suggestion"]
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO Feedback (name, email, phone, location, suggestion) VALUES (?, ?, ?, ?, ?)',
                           (name, email, phone, location, suggestion))
            conn.commit()
            flash("Thanks for your feedback!")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"Error: {e}")
        finally:
            conn.close()
    return render_template("feedback.html")

@app.route("/feedback_data")
def feedback_data():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Feedback')
    records = cursor.fetchall()
    conn.close()
    return render_template("feedback_data.html", feedback_records=records)

@app.route("/user_data")
def user_data():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Users')
    users = cursor.fetchall()
    conn.close()
    return render_template("user_data.html", user_records=users)

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form.get("user_input").lower()
    response = generate_response(user_input)
    return jsonify({"response": response})

@app.route("/about")
def about_us():
    return render_template("aboutus.html")

@app.route("/region")
def region():
    return render_template("region.html")

@app.route("/hospitals")
def hospitals():
    return render_template("hospitals.html")

@app.route("/appointments")
def appointments():
    return render_template("appointments.html")   

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")   

@app.route("/diet")
def diet():
    return render_template("diet.html")   

@app.route("/prescription_scanner")
def prescription_scanner():
    return render_template("new.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files["image"]
    if not image:
        return render_template("new.html", extracted_text="❌ No image uploaded.")

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    image_base64 = encode_image(filepath)
    response = make_gemini_request(image_base64)

    if response.status_code == 200:
        try:
            extracted_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            extracted_text = "❌ Failed to parse Gemini response."
    else:
        extracted_text = f"❌ API Error {response.status_code}: {response.text}"

    return render_template("new.html", extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)