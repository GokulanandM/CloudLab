from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mail import Mail, Message
from pymongo import MongoClient
import random
import string
import os
from datetime import datetime
from helperfunctions import (
    predict_disease_from_query, get_disease_info, get_doctors_for_disease,
     medical_preprocessor
)


app = Flask(__name__)
# Use environment variables for sensitive data
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_fallback_secret_key')

# Mapbox Token - Store securely (e.g., environment variable)
MAPBOX_ACCESS_TOKEN = os.environ.get('MAPBOX_ACCESS_TOKEN', 'pk.eyJ1Ijoia2FhbGFrYXJpa2FsYW4iLCJhIjoiY20yYjl2dzdqMHAydjJ3c2ZjYng1d2Q3YyJ9._iDR65eR_qkzbtrrcRjqag') # Your provided token


MONGO_URI = os.environ.get('MONGO_URI', "mongodb+srv://srikrish2705guru:krishguru05@cluster0.pgkpw0o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = 'marunthagam'
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database(DB_NAME)
    users = db.users
    # Test connection
    client.admin.command('ping')
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Fatal Error: Could not connect to MongoDB: {e}")
    exit()

# Mail config
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'gokulkvmhs2020@gmail.com'
app.config['MAIL_PASSWORD'] = 'jhmu ehci mkcs tdke'

mail = Mail(app)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# --- Helper Function ---
def save_message(username, text, sender):
    """Saves a message to the user's history in MongoDB."""
    if not username:
        print("Warning: Attempted to save message for non-logged-in user.")
        return False
    try:
        timestamp = datetime.now() # Store as datetime object
        users.update_one(
            {'username': username},
            {'$push': {'messages': {'text': text, 'timestamp': timestamp, 'sender': sender}}}
        )
        return True
    except Exception as e:
        print(f"Error saving message for user {username}: {e}")
        return False

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        re_password = request.form['re_password']

        if password != re_password:
            return "Passwords do not match"

        if users.find_one({'email': email}):
            return "Email already exists"

        users.insert_one({
            'username': username,
            'email': email,
            'password': password
        })
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email, 'password': password})
        if user:
            otp = ''.join(random.choices(string.digits, k=6))
            session['otp'] = otp
            session['user_email'] = email
            msg = Message('Your OTP for Marunthagam Login', sender='srikrish2705guru@gmail.com', recipients=[email])
            msg.body = f'Your OTP is {otp}'
            mail.send(msg)
            return redirect('/verify_otp')
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        entered_otp = request.form['otp']
        if entered_otp == session.get('otp'):
            user = users.find_one({'email': session['user_email']})
            session['username'] = user['username']
            return redirect('/chatbot')
        else:
            return "Invalid OTP"
    return render_template('verify_otp.html')


@app.route('/logout')
def logout():
     session.clear() # Clear the entire session
     flash('You have been logged out.', 'info')
     return redirect(url_for('login'))

@app.route('/chatbot')
def chatbot():
    if 'username' not in session:
        flash('Please login to access the chatbot.', 'warning')
        return redirect(url_for('login'))
    # Pass Mapbox token to the template
    return render_template('chatbot.html',
                           username=session['username'],
                           mapbox_token=MAPBOX_ACCESS_TOKEN)

# --- API Routes for Chatbot Interaction ---
@app.route('/process_message', methods=['POST'])
def process_message():
    if 'username' not in session:
        print("[DEBUG] User not logged in.")
        return jsonify({'error': 'User not logged in'}), 401

    data = request.get_json()
    user_message = data.get('message')
    username = session['username']
    print(f"[DEBUG] User message received: {user_message}")

    if not user_message:
        return jsonify({'reply': "Please type a message."})

    # Save user message to MongoDB
    users.update_one(
        {'username': username},
        {'$push': {'messages': {'text': user_message, 'timestamp': datetime.now(), 'sender': 'user'}}}
    )

    # NLP Processing
    processed_info = medical_preprocessor.process_query(user_message)
    print(f"[DEBUG] Processed info: {processed_info}")

    status = processed_info.get('status')
    intent = processed_info.get('intent')
    extracted_terms = processed_info.get('extracted_terms', [])
    has_negation = processed_info.get('has_negation', False)
    question_type = processed_info.get('question_type')

    # Default bot reply
    bot_reply = "Sorry, I couldn't understand that. Could you please rephrase?"
    follow_up_action = None

    if status == 'clarification_needed':
        similar_terms = processed_info.get('similar_terms', [])
        bot_reply = f"I didn't find any clear medical terms. Did you mean something like: {', '.join(similar_terms)}?"

    elif not extracted_terms:
        bot_reply = "Let's stick to medical topics. Could you describe a symptom, condition, or health-related issue?"

    elif status == 'processed':
        
        if question_type:
            term = extracted_terms[0] if extracted_terms else user_message
            bot_reply = medical_preprocessor.get_knowledge_base_response(question_type, term)
        # SYMPTOM INTENT
        elif intent == 'symptom':
            if has_negation:
                bot_reply = (
                    "Got it — you've mentioned symptoms you **don't** have. "
                    "Let me know what symptoms you **are** experiencing so I can assist further."
                )
            else:
                predicted_disease, found_symptoms = predict_disease_from_query(user_message)
                if predicted_disease != "Unknown Disease":
                    desc, diet, med, symps, prec, work = get_disease_info(predicted_disease)
                    bot_reply = (f"Based on your symptoms, it might be **{predicted_disease}**.\n\n"
                                 f"**Description:** {desc}\nWould you like to know more?")
                    session['last_predicted_disease'] = predicted_disease
                    follow_up_action = {
                        "type": "disease_info",
                        "disease": predicted_disease,
                        "options": ["Find Doctors", "Diet Info", "Medication Info", "Precautions", "Symptoms", "Workouts"]
                    }
                else:
                    bot_reply = "I couldn't predict a disease based on your symptoms. Could you provide more details?"

        # DISEASE INTENT
        elif intent == 'disease':
            disease_terms = [term for term in extracted_terms if term.lower() not in ['diet', 'medication', 'precautions', 'symptoms', 'workouts']]
            disease = disease_terms[0] if disease_terms else session.get('last_predicted_disease')
            session['last_predicted_disease'] = disease

            try:
                desc, diet, med, symps, prec, work = get_disease_info(disease)
                option_map = {
                    "diet": f"**Diet recommendations for {disease.title()}:**\n{diet}",
                    "medication": f"**Medications for {disease.title()}:**\n{med}",
                    "precautions": f"**Precautions for {disease.title()}:**\n" + "\n".join([f"- {p}" for p in prec]),
                    "symptoms": f"**Symptoms of {disease.title()}:**\n" + "\n".join([f"- {s}" for s in symps]),
                    "workouts": f"**Workouts for {disease.title()}:**\n" + "\n".join([f"- {w}" for w in work])
                }

                found = False
                for key in option_map:
                    if key in [term.lower() for term in extracted_terms]:
                        bot_reply = option_map[key]
                        found = True
                        break

                if not found:
                    bot_reply = f"Here is some information about **{disease}**:\n\n{desc}"

            except Exception as e:
                print(f"[ERROR] get_disease_info failed: {e}")
                bot_reply = "Sorry, something went wrong while fetching the details."

        # FIND DOCTOR INTENT
        elif intent == 'find_doctor':
            disease = session.get('last_predicted_disease')
            bot_reply = f"Please provide your location so I can find doctors for **{disease}**."
            

        # GENERAL QUESTION INTENT (e.g., what is, how to)
        elif question_type:
            term = extracted_terms[0] if extracted_terms else user_message
            bot_reply = medical_preprocessor.get_knowledge_base_response(question_type, term)

    # Save bot reply
    users.update_one(
        {'username': username},
        {'$push': {'messages': {'text': bot_reply, 'timestamp': datetime.now(), 'sender': 'bot'}}}
    )

    response = {'reply': bot_reply}
    if follow_up_action:
        response['follow_up'] = follow_up_action

    return jsonify(response)



@app.route('/get_doctors')
def get_doctors_api():
    disease = request.args.get('disease')
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    
    doctors_df = get_doctors_for_disease(disease)
    doctor_list = []
    
    for _, row in doctors_df.iterrows():
        try:
            doctor_list.append({
                "name": row["Doctor's Name"],
                "specialist": row["Specialist"],
                "lat": row["LAT"],
                "lon": row["LON"]
            })
        except:
            continue

    return jsonify({"doctors": doctor_list})


@app.route('/get_history')
def get_history():
    if 'username' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session['username']
    user_data = users.find_one({'username': username}, {'messages': 1, '_id': 0}) # Project only messages

    if user_data and 'messages' in user_data:
        # Convert datetime objects to strings for JSON serialization
        history = []
        for msg in user_data['messages']:
             # Ensure timestamp is serializable (ISO format string)
             ts = msg.get('timestamp')
             if isinstance(ts, datetime):
                 ts_str = ts.isoformat()
             elif isinstance(ts, str): # Already a string
                 ts_str = ts
             else: # Handle potential missing or wrong type timestamp
                 ts_str = datetime.now().isoformat() # Fallback

             history.append({
                'text': msg.get('text', ''),
                'sender': msg.get('sender', 'unknown'),
                'timestamp': ts_str
             })
        return jsonify(history)
    else:
        return jsonify([]) # Return empty list if no history
    
    

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure NLTK data is available (optional, can be done offline)
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/words')
    except LookupError:
        print("Downloading required NLTK data (punkt, averaged_perceptron_tagger, words)...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('words', quiet=True)

    # Run the Flask app
    # Use host='0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=8000) # Debug=True for development ONLY
