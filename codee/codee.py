import os
import pandas as pd
import numpy as np
import pickle
import spacy
import nltk
# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker') # Needed for some chunking patterns potentially
except nltk.downloader.DownloadError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words') # Needed for some chunking patterns potentially
except nltk.downloader.DownloadError:
    nltk.download('words')

from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk.tag import pos_tag
from flashtext import KeywordProcessor
import folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, flash
from datetime import datetime
import speech_recognition as sr
import json
import io

# --- Flask App Initialization ---
app = Flask(__name__)
# IMPORTANT: Set a secret key for session management
app.secret_key = os.urandom(24) # Or a fixed, strong secret key

# --- Load Data and Models (Similar to Streamlit setup) ---
def read_csv_with_encoding(file_path, encodings=['utf-8', 'iso-8859-1', 'cp1252']):
    for encoding in encodings:
        try:
            # Use os.path.join for better path handling
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            return pd.read_csv(full_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"Error: File not found at {full_path}")
            return None # Handle file not found case
    raise ValueError(f"Unable to read the file '{file_path}' with any of the provided encodings: {encodings}")

# --- Load Data ---
input_data = read_csv_with_encoding(r"datasets/Training.csv")
symptoms_severity_df = read_csv_with_encoding(r"datasets/Symptom-severity.csv")
description_df = read_csv_with_encoding(r"datasets/description.csv")
diets_df = read_csv_with_encoding(r"datasets/diets.csv")
medications_df = read_csv_with_encoding(r"datasets/medications.csv")
symptoms_df = read_csv_with_encoding(r"datasets/symtoms_df.csv") # Corrected path if filename is indeed 'symtoms_df.csv'
precautions_df = read_csv_with_encoding(r"datasets/precautions_df.csv")
workouts_df = read_csv_with_encoding(r"datasets/workout_df.csv")
doctors_df = read_csv_with_encoding(r"datasets/doc.csv")

# Handle potential loading errors
if input_data is None or any(df is None for df in [symptoms_severity_df, description_df, diets_df, medications_df, symptoms_df, precautions_df, workouts_df, doctors_df]):
    print("Error loading one or more data files. Exiting.")
    exit()


# --- Load Model ---
try:
    model_path = os.path.join(os.path.dirname(__file__), r"models/gbm_model.pkl")
    with open(model_path, "rb") as file:
        gbm = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- Setup Features and Keyword Processor ---
features = input_data.columns[:-1]
prognosis = input_data['prognosis']
feature_dict = {feature.replace('_', ' '): idx for idx, feature in enumerate(features)} # Normalize keys
keyword_processor = KeywordProcessor(case_sensitive=False) # Make case-insensitive
keyword_processor.add_keywords_from_list(list(feature_dict.keys()))

# --- Initialize Medical Preprocessor ---
class MedicalPreprocessor:
    def __init__(self, diseases, symptoms):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
            exit()
        self.diseases = [str(d).lower().strip() for d in diseases if pd.notna(d)]
        self.symptoms = [str(s).lower().strip().replace('_', ' ') for s in symptoms if pd.notna(s)] # Normalize symptoms
        self.chunk_grammar = r"""
            Medical: {<JJ.*>*<NN.*>+}
            Symptom: {<VB.*>?<JJ.*>*<NN.*>+}
        """
        self.chunk_parser = RegexpParser(self.chunk_grammar)

    def normalize_text(self, text):
        return str(text).lower().strip() # Ensure input is string

    def find_matches(self, text):
        text_norm = self.normalize_text(text)
        # Use keyword processor for efficient symptom matching
        symptom_matches = keyword_processor.extract_keywords(text_norm)
        # Check for disease names (simple substring check for now)
        disease_matches = [d for d in self.diseases if d in text_norm]
        return disease_matches, symptom_matches

    def determine_intent(self, text, disease_matches, symptom_matches):
        text_norm = self.normalize_text(text)
        symptom_indicators = {'symptom', 'symptoms', 'experiencing', 'feel', 'feeling', 'suffer', 'suffering', 'pain', 'ache'}
        disease_indicators = {'do i have', 'is this', 'diagnosed', 'disease', 'condition', 'illness'}

        has_symptom_indicator = any(indicator in text_norm for indicator in symptom_indicators)
        has_disease_indicator = any(indicator in text_norm for indicator in disease_indicators)

        if "symptoms of" in text_norm and disease_matches:
             return 'disease_symptom_query' # Specific intent for "symptoms of X"

        if symptom_matches or has_symptom_indicator:
             return 'symptom' # Prioritize symptom intent if symptoms or indicators are present

        if disease_matches or has_disease_indicator:
             return 'disease'

        # Default guess based on content if no strong indicators
        if disease_matches and not symptom_matches:
             return 'disease'
        if symptom_matches and not disease_matches:
             return 'symptom'

        return 'unknown' # Or default to 'symptom' if unsure

    def process_query(self, query):
        normalized_query = self.normalize_text(query)
        disease_matches, symptom_matches = self.find_matches(normalized_query)
        intent = self.determine_intent(normalized_query, disease_matches, symptom_matches)

        extracted = []
        if intent == 'disease_symptom_query':
             extracted = disease_matches # Focus on the disease mentioned
             intent = 'disease' # Treat it as asking about a disease contextually
        elif intent == 'symptom':
             extracted = symptom_matches
        elif intent == 'disease':
             extracted = disease_matches
        else: # unknown or fallback
             # Try extracting both if intent is unclear
             extracted = list(set(disease_matches + symptom_matches))
             # If still nothing, maybe just use the symptom matches if any
             if not extracted and symptom_matches:
                 extracted = symptom_matches
                 intent = 'symptom' # Reclassify based on found terms


        # If intent is symptom but no specific symptoms extracted, maybe the query *is* the symptom
        if intent == 'symptom' and not extracted and query:
             # Basic check if query looks like a symptom (e.g., noun phrase)
             # This is simplistic and could be improved
             doc = self.nlp(normalized_query)
             if any(token.pos_ in ['NOUN', 'PROPN', 'ADJ'] for token in doc):
                extracted = [normalized_query] # Use the whole query


        print(f"DEBUG: Query='{query}', Normalized='{normalized_query}', Intent='{intent}', Diseases='{disease_matches}', Symptoms='{symptom_matches}', Extracted='{extracted}'")
        return {'intent': intent, 'extracted': extracted}


# --- Helper Functions (Adapted for Flask context) ---
def predict_disease_from_symptoms(selected_symptoms):
    sample_x = np.zeros(len(features))
    matched_count = 0
    for symptom in selected_symptoms:
        # Use the normalized symptom name from feature_dict
        normalized_symptom = symptom.lower().strip()
        if normalized_symptom in feature_dict:
            sample_x[feature_dict[normalized_symptom]] = 1
            matched_count += 1
        else:
             print(f"Warning: Symptom '{symptom}' not found in feature dictionary.")

    if matched_count == 0:
        print("Warning: No matched symptoms found for prediction.")
        return "Could not determine disease (no matching symptoms)", selected_symptoms

    sample_x = sample_x.reshape(1, -1)
    predicted_index = gbm.predict(sample_x)[0] # gbm should predict index

    # Ensure prognosis is indexed correctly (assuming it aligns with model output)
    # Need to map the predicted index back to the disease name.
    # Assuming the model predicts the index corresponding to the order in prognosis.unique()
    unique_prognosis = input_data['prognosis'].astype('category')
    le = {i: cat for i, cat in enumerate(unique_prognosis.cat.categories)}

    if predicted_index in le:
         predicted_disease = le[predicted_index]
    else:
         # Fallback or error handling if index is unexpected
         print(f"Error: Predicted index {predicted_index} out of bounds for prognosis mapping.")
         predicted_disease = "Unknown Disease Prediction"

    return predicted_disease, selected_symptoms

def get_distinct_symptoms(disease):
    disease_lower = str(disease).lower().strip()
    # Ensure 'Disease' column exists and handle potential case mismatches
    if 'Disease' not in symptoms_df.columns:
        print("Error: 'Disease' column not found in symptoms_df.")
        return []
    disease_symptoms = symptoms_df[symptoms_df['Disease'].str.lower().str.strip() == disease_lower]
    if disease_symptoms.empty:
        print(f"No symptoms found for disease: '{disease}'")
        return []
    # Get all columns except 'Disease'
    symptom_columns = [col for col in disease_symptoms.columns if col.lower() != 'disease']
    all_symptoms = disease_symptoms[symptom_columns].values.flatten()
    # Clean and get unique symptoms
    distinct_symptoms = sorted(list(set([str(symptom).strip() for symptom in all_symptoms if pd.notna(symptom) and str(symptom).strip()])))
    return distinct_symptoms


def get_disease_info(disease):
    disease_lower = str(disease).lower().strip() # Ensure lowercase and string

    # --- Description ---
    description_row = description_df[description_df['Disease'].str.lower().str.strip() == disease_lower]
    if description_row.empty:
        print(f"Disease '{disease}' not found in description_df")
        # Return None or default values for all pieces of information
        return None, None, None, [], [], []
    description = description_row['Description'].iloc[0]

    # --- Diet ---
    diet_row = diets_df[diets_df['Disease'].str.lower().str.strip() == disease_lower]
    # Handle lists stored as strings: '["item1", "item2"]' -> ["item1", "item2"]
    diet_info = "No specific diet information found."
    if not diet_row.empty:
        diet_str = diet_row['Diet'].iloc[0]
        try:
            # Attempt to parse if it looks like a list string
            if isinstance(diet_str, str) and diet_str.startswith('[') and diet_str.endswith(']'):
                 # Use json.loads for safety over eval
                 parsed_diet = json.loads(diet_str.replace("'", '"')) # Replace single quotes for valid JSON
                 diet_info = ", ".join(parsed_diet) if isinstance(parsed_diet, list) else diet_str
            else:
                 diet_info = diet_str # Assume it's a plain string if not list-like
        except (json.JSONDecodeError, TypeError):
             diet_info = diet_str # Fallback to original string on parsing error

    # --- Medication ---
    med_row = medications_df[medications_df['Disease'].str.lower().str.strip() == disease_lower]
    medication_info = "No specific medication information found."
    if not med_row.empty:
        med_str = med_row['Medication'].iloc[0]
        try:
            if isinstance(med_str, str) and med_str.startswith('[') and med_str.endswith(']'):
                 parsed_med = json.loads(med_str.replace("'", '"'))
                 medication_info = ", ".join(parsed_med) if isinstance(parsed_med, list) else med_str
            else:
                 medication_info = med_str
        except (json.JSONDecodeError, TypeError):
             medication_info = med_str

    # --- Distinct Symptoms (using the corrected function) ---
    distinct_symptoms = get_distinct_symptoms(disease)

    # --- Precautions ---
    prec_row = precautions_df[precautions_df['Disease'].str.lower().str.strip() == disease_lower]
    precautions = []
    if not prec_row.empty:
        # Assuming precautions are in columns like 'Precaution_1', 'Precaution_2', etc.
        prec_cols = [col for col in prec_row.columns if col.lower().startswith('precaution')]
        prec_values = prec_row[prec_cols].values.flatten()
        precautions = sorted(list(set([str(p).strip() for p in prec_values if pd.notna(p) and str(p).strip()])))

    # --- Workouts ---
    workout_row = workouts_df[workouts_df['disease'].str.lower().str.strip() == disease_lower]
    workouts = []
    if not workout_row.empty:
        # Assuming workouts are in a single column 'workout' potentially containing list-like strings or multiple rows
        workout_values = workout_row['workout'].values
        temp_workouts = []
        for w in workout_values:
             if pd.notna(w):
                 w_str = str(w).strip()
                 if w_str.startswith('[') and w_str.endswith(']'):
                     try:
                         parsed_w = json.loads(w_str.replace("'", '"'))
                         if isinstance(parsed_w, list):
                             temp_workouts.extend([item.strip() for item in parsed_w])
                         else:
                             temp_workouts.append(str(parsed_w).strip()) # Add as single item if not list
                     except (json.JSONDecodeError, TypeError):
                          temp_workouts.append(w_str) # Fallback
                 else:
                      temp_workouts.append(w_str)
        workouts = sorted(list(set(temp_workouts)))


    return description, diet_info, medication_info, distinct_symptoms, precautions, workouts


def get_color_for_severity(weight):
    try:
        weight = int(weight)
        if weight <= 2: return 'green'
        elif weight <= 4: return 'orange' # Changed yellow to orange for better visibility
        else: return 'red'
    except (ValueError, TypeError):
        return 'grey' # Default color if weight is not a valid number

# Initialize NLP Preprocessor
diseases_list = description_df['Disease'].unique().tolist() # Use diseases from description df
symptoms_list = list(feature_dict.keys())
medical_preprocessor = MedicalPreprocessor(diseases_list, symptoms_list)


def get_doctors_for_disease(disease):
    disease_lower = str(disease).lower().strip()
    # Ensure required columns exist and handle case/whitespace
    if not all(col in doctors_df.columns for col in ['Disease', 'LAT', 'LON', "Doctor's Name", 'Specialist', 'ADDRESS']):
         print("Error: Required columns missing in doctors_df.")
         return pd.DataFrame() # Return empty dataframe

    doctors_df['Disease_lower'] = doctors_df['Disease'].str.lower().str.strip()

    # Exact match first
    doctors = doctors_df[doctors_df['Disease_lower'] == disease_lower]

    if doctors.empty: # Partial match if no exact found
        doctors = doctors_df[doctors_df['Disease_lower'].str.contains(disease_lower, na=False)] # Handle NaN

    # Drop the temporary column
    doctors_df.drop(columns=['Disease_lower'], inplace=True, errors='ignore')
    doctors = doctors.copy() # Avoid SettingWithCopyWarning

    # Ensure LAT/LON are numeric
    doctors['LAT'] = pd.to_numeric(doctors['LAT'], errors='coerce')
    doctors['LON'] = pd.to_numeric(doctors['LON'], errors='coerce')
    doctors.dropna(subset=['LAT', 'LON'], inplace=True) # Remove doctors with invalid coordinates

    print(f"Found {len(doctors)} doctors for disease '{disease}'.")
    return doctors[['Doctor\'s Name', 'Specialist', 'ADDRESS', 'LAT', 'LON']] # Return specific columns

def calculate_travel_time(start_lat, start_lon, end_lat, end_lon):
    if None in [start_lat, start_lon, end_lat, end_lon]:
        return "N/A (Missing Coordinates)"
    try:
        distance = geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
        # Simple model: average speed 40 km/h
        time_hours = distance / 40
        if time_hours < 1:
            return f"{time_hours * 60:.0f} minutes"
        else:
            return f"{time_hours:.1f} hours"
    except Exception as e:
        print(f"Error calculating travel time: {e}")
        return "N/A (Calculation Error)"

def get_lat_lon(address):
    try:
        geolocator = Nominatim(user_agent="marunthagam_med_assistant") # Use a specific user agent
        location = geolocator.geocode(address, timeout=10) # Add timeout
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error for address '{address}': {e}")
        return None, None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize session variables if not present
    if 'messages' not in session:
        session['messages'] = []
    if 'query_history' not in session:
        session['query_history'] = []
    if 'user_lat' not in session:
        session['user_lat'] = 10.0 # Default Lat (e.g., somewhere in India)
    if 'user_lon' not in session:
        session['user_lon'] = 78.0 # Default Lon
    if 'current_disease' not in session:
        session['current_disease'] = None
    if 'disease_info_cache' not in session:
        session['disease_info_cache'] = {} # Cache disease info
    if 'show_options' not in session:
        session['show_options'] = False
    if 'selected_option' not in session:
        session['selected_option'] = None
    if 'map_html' not in session:
        session['map_html'] = None
    if 'doctor_details' not in session:
        session['doctor_details'] = None


    # --- Handle POST requests ---
    if request.method == 'POST':
        form_type = request.form.get('form_type') # Hidden input to identify form

        if form_type == 'location_update':
            address = request.form.get('address')
            manual_lat = request.form.get('manual_lat')
            manual_lon = request.form.get('manual_lon')

            if address:
                lat, lon = get_lat_lon(address)
                if lat and lon:
                    session['user_lat'] = lat
                    session['user_lon'] = lon
                    flash(f"Location updated to {address} ({lat:.4f}, {lon:.4f})", "success")
                else:
                    flash(f"Could not find coordinates for '{address}'. Please check the address or enter manually.", "warning")
            elif manual_lat and manual_lon:
                try:
                    session['user_lat'] = float(manual_lat)
                    session['user_lon'] = float(manual_lon)
                    flash(f"Location updated manually to ({manual_lat}, {manual_lon})", "success")
                except ValueError:
                    flash("Invalid latitude or longitude provided.", "error")
            session.modified = True # Ensure session updates are saved
            # No redirect needed if updating via AJAX, otherwise:
            # return redirect(url_for('index'))

        elif form_type == 'chat_input':
            query = request.form.get('query')
            if query:
                session['query_history'].append(f"{datetime.now().strftime('%H:%M')}: {query}")
                session['messages'].append({"role": "user", "content": query})

                # Reset state for new query
                session['show_options'] = False
                session['current_disease'] = None
                session['selected_option'] = None
                session['map_html'] = None
                session['doctor_details'] = None


                processed_result = medical_preprocessor.process_query(query)
                intent = processed_result['intent']
                extracted_terms = processed_result['extracted']

                # Basic assistant response about processing
                nlp_info = f"Understood. You mentioned: {', '.join(extracted_terms)}. Analyzing..." if extracted_terms else "Processing your query..."
                session['messages'].append({"role": "assistant", "content": nlp_info})


                if intent == 'symptom' and extracted_terms:
                    predicted_disease, selected_symptoms = predict_disease_from_symptoms(extracted_terms)
                    session['current_disease'] = predicted_disease
                    session['messages'].append({"role": "assistant", "content": f"Based on the symptoms, it might be related to: **{predicted_disease}**."})

                    # Display symptom severities if applicable
                    severity_message = "### Symptom Analysis:\n"
                    has_severity = False
                    for symptom in selected_symptoms:
                         symptom_norm = symptom.lower().strip()
                         severity_row = symptoms_severity_df[symptoms_severity_df['Symptom'].str.lower().str.strip() == symptom_norm]
                         if not severity_row.empty:
                              weight = severity_row['weight'].iloc[0]
                              color = get_color_for_severity(weight)
                              severity_message += f"- <span style='color:{color};'>{symptom.capitalize()} (Severity: {weight})</span>\n"
                              has_severity = True
                         else:
                              severity_message += f"- {symptom.capitalize()} (Severity info not available)\n"
                    if has_severity:
                        session['messages'].append({"role": "assistant", "content": severity_message})

                    # Fetch and display description
                    if predicted_disease != "Could not determine disease (no matching symptoms)":
                        description, _, _, _, _, _ = get_disease_info(predicted_disease)
                        if description:
                            session['messages'].append({"role": "assistant", "content": f"### About {predicted_disease}\n{description}"})
                        else:
                             session['messages'].append({"role": "assistant", "content": f"Could not find a description for {predicted_disease}."})
                        session['show_options'] = True # Show options if disease identified
                    else:
                        session['messages'].append({"role": "assistant", "content": "Please provide more specific symptoms for a clearer prediction."})


                elif intent == 'disease' and extracted_terms:
                    disease_name = extracted_terms[0].capitalize() # Use the first matched disease
                    session['current_disease'] = disease_name
                    description, _, _, _, _, _ = get_disease_info(disease_name)
                    if description:
                        session['messages'].append({"role": "assistant", "content": f"### About {disease_name}\n{description}"})
                        session['show_options'] = True # Show options if disease found
                    else:
                        session['messages'].append({"role": "assistant", "content": f"Sorry, I couldn't find detailed information for '{disease_name}'."})

                else: # Fallback or unknown intent
                    session['messages'].append({"role": "assistant", "content": "I couldn't determine a specific disease from your query. Can you please describe your symptoms or the condition you're asking about more clearly?"})


                session.modified = True
                # return redirect(url_for('index')) # Reload page to show updates


        elif form_type == 'options_form':
             option = request.form.get('option')
             if option and session.get('current_disease'):
                 session['selected_option'] = option
                 current_disease = session['current_disease']

                 # Fetch disease info if not cached
                 if current_disease not in session['disease_info_cache']:
                      session['disease_info_cache'][current_disease] = get_disease_info(current_disease)

                 disease_info = session['disease_info_cache'].get(current_disease)

                 if not disease_info or disease_info[0] is None: # Check if info retrieval failed
                     session['messages'].append({"role": "assistant", "content": f"Sorry, detailed information for {current_disease} is currently unavailable."})

                 else:
                     description, diet, medication, distinct_symptoms, precautions, workouts = disease_info

                     option_map = {
                         "Recommended Diet": diet,
                         "Medications": medication,
                         "Other Symptoms": distinct_symptoms,
                         "Precautions": precautions,
                         "Suggested Workouts": workouts
                     }

                     if option == "exit(to end)":
                         session['messages'].append({"role": "assistant", "content": "Thank you for using Marunthagam Medical Assistant. Feel free to ask more questions later. Goodbye!"})
                         session['show_options'] = False
                         session['current_disease'] = None # Clear current disease context
                         # Optionally add feedback buttons/logic here

                     elif option == "Show recommended doctors":
                         doctors = get_doctors_for_disease(current_disease)
                         session['doctor_details'] = [] # Reset details
                         if not doctors.empty:
                             session['messages'].append({"role": "assistant", "content": f"### Recommended Doctors for {current_disease}\nBased on your location ({session['user_lat']:.4f}, {session['user_lon']:.4f}):"})

                             m = folium.Map(location=[session['user_lat'], session['user_lon']], zoom_start=10)
                             folium.Marker(
                                 [session['user_lat'], session['user_lon']],
                                 popup="Your Location",
                                 tooltip="Your Location",
                                 icon=folium.Icon(color="red", icon="user", prefix='fa'),
                             ).add_to(m)

                             doc_list_html = "<ul>"
                             for index, doctor in doctors.iterrows():
                                 doc_lat, doc_lon = doctor['LAT'], doctor['LON']
                                 doc_name = doctor["Doctor's Name"]
                                 doc_spec = doctor['Specialist']
                                 doc_addr = doctor['ADDRESS']
                                 travel_time = calculate_travel_time(session['user_lat'], session['user_lon'], doc_lat, doc_lon)

                                 popup_html = f"""
                                 <b>{doc_name}</b><br>
                                 <i>{doc_spec}</i><br>
                                 Address: {doc_addr}<br>
                                 Est. Travel Time: {travel_time}
                                 """
                                 folium.Marker(
                                     [doc_lat, doc_lon],
                                     popup=folium.Popup(popup_html, max_width=300),
                                     tooltip=f"{doc_name} ({travel_time})",
                                     icon=folium.Icon(color="green", icon="plus-square", prefix='fa'), # Font Awesome icons
                                 ).add_to(m)
                                 doc_list_html += f"<li><b>{doc_name}</b> ({doc_spec}) - {doc_addr} | Approx. Travel: {travel_time}</li>"
                                 session['doctor_details'].append({
                                     'name': doc_name, 'specialist': doc_spec, 'address': doc_addr, 'travel_time': travel_time
                                 })


                             doc_list_html += "</ul>"
                             session['messages'].append({"role": "assistant", "content": doc_list_html})

                             # Save map HTML to session
                             map_html = m._repr_html_()
                             session['map_html'] = map_html
                         else:
                             session['messages'].append({"role": "assistant", "content": f"Sorry, no doctors found specializing in {current_disease} in the database."})
                             session['map_html'] = None # Clear map if no doctors
                             session['doctor_details'] = None

                     elif option in option_map:
                         info = option_map[option]
                         info_message = f"### {option} for {current_disease}\n"
                         if isinstance(info, list) and info:
                             info_message += "<ul>" + "".join([f"<li>{item}</li>" for item in info]) + "</ul>"
                         elif isinstance(info, str) and info:
                             info_message += info
                         else:
                             info_message += f"No specific information found for {option} regarding {current_disease}."
                         session['messages'].append({"role": "assistant", "content": info_message})

                     else: # Fallback for unexpected option
                         session['messages'].append({"role": "assistant", "content": "Sorry, I couldn't process that option."})

             session.modified = True
             # return redirect(url_for('index')) # Reload page


    # --- Handle GET requests (Render the page) ---
    # Get user info from query params ONLY on initial load maybe? Or handle login separately.
    user_name = request.args.get("name", session.get("user_name", "")) # Persist in session
    user_email = request.args.get("email", session.get("user_email", ""))
    if user_name and 'user_name' not in session: session['user_name'] = user_name
    if user_email and 'user_email' not in session: session['user_email'] = user_email


    return render_template('index.html',
                           messages=session['messages'],
                           query_history=session['query_history'],
                           user_name=session.get('user_name'),
                           user_email=session.get('user_email'),
                           user_lat=session['user_lat'],
                           user_lon=session['user_lon'],
                           current_disease=session.get('current_disease'),
                           show_options=session.get('show_options'),
                           selected_option=session.get('selected_option'),
                           map_html=session.get('map_html'),
                           doctor_details=session.get('doctor_details')
                           )


# --- Route for Speech Recognition (Called by JavaScript) ---
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files['audio_data']

    # Use SpeechRecognition to process the audio file
    r = sr.Recognizer()
    try:
        # Read the file-like object directly
        with io.BytesIO(audio_file.read()) as source_data:
             # We need to convert the raw bytes (likely webm/ogg from browser)
             # to a format SR understands (like WAV). This is the tricky part.
             # For simplicity here, we'll assume the JS sends WAV,
             # or use a library like pydub if conversion is needed.
             # Let's *assume* it's WAV for now, which might fail.
             # A more robust solution involves ffmpeg or pydub.

             # Simplistic approach (might need conversion):
             # Create an AudioFile object from the BytesIO stream
             with sr.AudioFile(source_data) as source:
                audio_data = r.record(source) # Read the entire audio file

        # Recognize speech using Google Web Speech API
        text = r.recognize_google(audio_data)
        print(f"Speech recognized: {text}")
        return jsonify({"text": text})

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return jsonify({"error": f"Speech service error: {e}"}), 500
    except Exception as e:
        # Catch potential issues with audio format
        print(f"Error processing audio file: {e}")
        return jsonify({"error": f"Server error processing audio: {e}"}), 500


# --- Logout Route ---
@app.route('/logout')
def logout():
    session.clear() # Clear all session data
    flash("You have been logged out.", "info")
    # Redirect to a hypothetical login page or the main page
    # Assuming the original target 'http://127.0.0.1:5000/' is the login/entry point
    return redirect("http://127.0.0.1:5000/") # Or redirect(url_for('index')) if no separate login


# --- Run the App ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network
    # Use a specific port if needed
    app.run(debug=True, host='0.0.0.0', port=5001) # Run on a different port than 5000 if that's the login app