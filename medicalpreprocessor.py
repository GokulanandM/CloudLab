import pandas as pd
import spacy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk.tag import pos_tag
from flashtext import KeywordProcessor
from spellchecker import SpellChecker
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

class EnhancedMedicalPreprocessor:
    def __init__(self, diseases, symptoms, kb_path='datasets/kb.csv'):
        self.nlp = spacy.load('en_core_web_sm')
        self.diseases = [d.lower() for d in diseases]
        self.symptoms = [s.lower() for s in symptoms]
        self.kb_data = pd.read_csv(kb_path)
        self.medical_terms = list(set(self.kb_data['Symptoms'].dropna().tolist() + self.kb_data['Diseases'].dropna().tolist()))
        self.vectorizer = TfidfVectorizer()
        self.term_vectors = self.vectorizer.fit_transform(self.medical_terms)
        self.chunk_grammar = r"""
            Medical: {<JJ.*>*<NN.*>+}
            Symptom: {<VB.*>?<JJ.*>*<NN.*>+}
            Question: {<W.*>|<MD>}
            NegPhrase: {<RB>*<NOT>|<RB>*<DT>*<VB.*>}
        """
        self.chunk_parser = RegexpParser(self.chunk_grammar)
        self.keyword_processor = KeywordProcessor()
        self.keyword_processor.add_keywords_from_list(self.medical_terms)

    def find_similar_terms(self, term, threshold=0.5):
        spell = SpellChecker()
        corrected_term = spell.correction(term)
        term_vector = self.vectorizer.transform([corrected_term])
        similarities = cosine_similarity(term_vector, self.term_vectors).flatten()
        similar_indices = similarities.argsort()[-5:][::-1]
        for idx in similar_indices:
            if similarities[idx] >= threshold:
                return self.medical_terms[idx]
        leven_matches = get_close_matches(corrected_term, self.medical_terms, n=1, cutoff=0.7)
        return leven_matches[0] if leven_matches else corrected_term

    def extract_medical_entities(self, text):
        doc = self.nlp(text)
        medical_entities = [ent.text for ent in doc.ents if ent.label_ in ['DISEASE', 'SYMPTOM']]
        medical_entities.extend(self.keyword_processor.extract_keywords(text))
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = self.chunk_parser.parse(pos_tags)
        for subtree in chunks.subtrees(filter=lambda t: t.label() in ['Medical', 'Symptom']):
            entity = ' '.join([word for word, tag in subtree.leaves()])
            medical_entities.append(entity)
        return list(set(medical_entities))

    def check_negation(self, text):
        return any(token.dep_ == 'neg' for token in self.nlp(text))

    def detect_question_type(self, text):
        doc = self.nlp(text)
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}
        first_word = doc[0].text.lower()
        if first_word in question_words or text.endswith('?'):
            if any(word.text.lower() in {'what', 'how'} for word in doc):
                return 'information'
            if any(word.text.lower() in {'is', 'are', 'do', 'does'} for word in doc):
                return 'verification'
            return 'general'
        return None

    def process_query(self, query):
        normalized_query = query.lower().strip()
        medical_entities = self.extract_medical_entities(normalized_query)
        if not medical_entities:
            for word in word_tokenize(normalized_query):
                similar_terms = self.find_similar_terms(word)
                if similar_terms:
                    return {'status': 'clarification_needed', 'similar_terms': similar_terms, 'original_term': word}
        has_negation = self.check_negation(normalized_query)
        question_type = self.detect_question_type(normalized_query)
        intent = self.determine_intent(normalized_query, medical_entities)
        sentiment = TextBlob(normalized_query).sentiment.polarity
        return {
            'status': 'processed',
            'intent': intent,
            'extracted_terms': medical_entities,
            'has_negation': has_negation,
            'question_type': question_type,
            'sentiment': sentiment
        }

    def determine_intent(self, text, extracted_terms):
        symptom_indicators = {'symptom', 'symptoms', 'experiencing', 'feel', 'feeling', 'suffer', 'suffering'}
        disease_indicators = {'do i have', 'is this', 'diagnosed', 'disease', 'condition'}
        disease_matches = [term for term in extracted_terms if term.lower() in self.diseases]
        symptom_matches = [term for term in extracted_terms if term.lower() in self.symptoms]
        if any(indicator in text for indicator in disease_indicators) or disease_matches:
            return 'disease'
        if any(indicator in text for indicator in symptom_indicators) or symptom_matches:
            return 'symptom'
        return 'disease' if disease_matches and not symptom_matches else 'symptom'

    def get_knowledge_base_response(self, query_type, term):
        if query_type == 'information':
            matches = self.kb_data[
            (self.kb_data['Symptoms'].str.contains(term, na=False, case=False)) |
            (self.kb_data['Diseases'].str.contains(term, na=False, case=False))
        ]
        
            if not matches.empty:
            # Groq API Query
                apiKey = "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"
                groqEndpoint = "https://api.groq.com/openai/v1/chat/completions"
            
            # Prepare the Groq API request payload
                data = {
                "model": "llama-3.3-70b-versatile", # Adjust the model as necessary
                "messages": [{"role": "user", "content": f"Describe {term} medically."}]
            }
            
                headers = {
                "Authorization": f"Bearer {apiKey}",
                "Content-Type": "application/json"
            }
            
            # Send the request to GroqAPI
                response = requests.post(groqEndpoint, json=data, headers=headers)
            
                if response.status_code == 200:
                    response_data = response.json()
                # Assuming the response is a JSON with a 'choices' field containing the answer
                    description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if description:
                        return f"Description found on GroqAPI: {description}"
                    else:
                        return f"No relevant information found on GroqAPI for {term}."
                else:
                    return f"Failed to fetch data from GroqAPI. Status Code: {response.status_code}"
        
            return f"No matching symptoms or diseases found in the knowledge base for {term}."
        return f"{term} not found in the medical knowledge base."