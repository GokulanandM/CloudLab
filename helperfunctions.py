import pandas as pd
import numpy as np
import pickle
from flashtext import KeywordProcessor
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from medicalpreprocessor import EnhancedMedicalPreprocessor


def read_csv_with_encoding(file_path, encodings=['utf-8', 'iso-8859-1', 'cp1252']):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file with any of the provided encodings: {encodings}")

import os

# Use os.path.join for cross-platform compatibility
dataset_folder = 'datasets'

input_data = read_csv_with_encoding(os.path.join(dataset_folder, 'Training.csv'))
with open(os.path.join('models', 'gbm_model.pkl'), 'rb') as file:
    gbm = pickle.load(file)

# Load datasets
symptoms_severity_df = read_csv_with_encoding(os.path.join(dataset_folder, 'Symptom-severity.csv'))
description_df = read_csv_with_encoding(os.path.join(dataset_folder, 'description.csv'))
diets_df = read_csv_with_encoding(os.path.join(dataset_folder, 'diets.csv'))
medications_df = read_csv_with_encoding(os.path.join(dataset_folder, 'medications.csv'))
symptoms_df = read_csv_with_encoding(os.path.join(dataset_folder, 'symtoms_df.csv'))
precautions_df = read_csv_with_encoding(os.path.join(dataset_folder, 'precautions_df.csv'))
workouts_df = read_csv_with_encoding(os.path.join(dataset_folder, 'workout_df.csv'))
doctors_df = read_csv_with_encoding(os.path.join(dataset_folder, 'doc.csv'))

# Initialize features and keyword processor
features = input_data.columns[:-1]
prognosis = input_data['prognosis']
feature_dict = {feature: idx for idx, feature in enumerate(features)}
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(features.tolist())

def predict_disease_from_query(query):
    selected_symptoms = keyword_processor.extract_keywords(query)
    sample_x = np.zeros(len(features))
    for symptom in selected_symptoms:
        if symptom in feature_dict:
            sample_x[feature_dict[symptom]] = 1
    sample_x = sample_x.reshape(1, -1)
    predicted_result = gbm.predict(sample_x)[0]
    if isinstance(predicted_result, str):
        predicted_disease = predicted_result
    else:
        predicted_index = int(predicted_result)
        predicted_disease = prognosis.iloc[predicted_index] if predicted_index < len(prognosis) else "Unknown Disease"
    return predicted_disease, selected_symptoms

def get_distinct_symptoms(disease):
    disease_symptoms = symptoms_df[symptoms_df['Disease'].str.lower() == disease.lower()]
    if disease_symptoms.empty:
        return []
    symptom_columns = disease_symptoms.columns[1:]
    all_symptoms = disease_symptoms[symptom_columns].values.flatten()
    return list(set([str(symptom).strip() for symptom in all_symptoms if pd.notna(symptom)]))

def get_disease_info(disease):
    disease_lower = disease.lower()
    disease_row = description_df[description_df['Disease'].str.lower() == disease_lower]
    if disease_row.empty:
        raise ValueError(f"No information found for disease '{disease}'")
    description = disease_row['Description'].values[0]
    diet = diets_df[diets_df['Disease'].str.lower() == disease_lower]['Diet'].values[0]
    medication = medications_df[diets_df['Disease'].str.lower() == disease_lower]['Medication'].values[0]
    distinct_symptoms = get_distinct_symptoms(disease)
    precautions = precautions_df[precautions_df['Disease'].str.lower() == disease_lower].drop('Disease', axis=1).values.flatten()
    precautions = list(set([str(precaution) for precaution in precautions if pd.notna(precaution)]))
    workouts = workouts_df[workouts_df['disease'].str.lower() == disease_lower]['workout'].values
    workouts = list(set([str(workout) for workout in workouts if pd.notna(workout)]))
    return description, diet, medication, distinct_symptoms, precautions, workouts

def get_color_for_severity(weight):
    return 'green' if weight <= 2 else 'yellow' if weight <= 4 else 'red'

diseases_list = prognosis.unique().tolist()
symptoms_list = features.tolist()
medical_preprocessor = EnhancedMedicalPreprocessor(diseases_list, symptoms_list)

def get_doctors_for_disease(disease):
    disease_lower = disease.lower().strip()
    doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip() == disease_lower]
    if doctors.empty:
        doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip().str.contains(disease_lower)]
    if doctors.empty:
        disease_words = set(disease_lower.split())
        doctors = doctors_df[doctors_df['Disease'].str.lower().str.strip().apply(lambda x: set(x.split()).intersection(disease_words))]
    return doctors

def calculate_travel_time(start_lat, start_lon, end_lat, end_lon):
    distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
    return f"{distance / 40:.2f} hours"

def get_lat_lon(address):
    geolocator = Nominatim(user_agent="geoapi")
    location = geolocator.geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)
# Re-export symptoms list (already passed to EnhancedMedicalPreprocessor)
def get_all_symptoms():
    return symptoms_list
