# models.py - Machine learning models for legal prediction
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class LegalOutcomePredictor:
    def __init__(self):
        """Initialize the outcome prediction model"""
        self.model = None
        self.trained = False
    
    def train(self, case_texts, outcomes):
        """Train the outcome prediction model"""
        # Create a pipeline with TF-IDF and RandomForest
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            case_texts, outcomes, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        self.trained = True
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def predict(self, case_text):
        """Predict outcome for a given case text"""
        if not self.trained or self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get probability predictions
        proba = self.model.predict_proba([case_text])[0]
        
        # Get class with highest probability
        pred_class = self.model.predict([case_text])[0]
        
        return {
            'predicted_outcome': pred_class,
            'confidence': float(np.max(proba)),
            'class_probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.model.classes_, proba)
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if not self.trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, filepath)
        return True
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.model = joblib.load(filepath)
        self.trained = True
        return True

class LegalEntityRecognizer:
    def __init__(self):
        """Initialize custom legal entity recognizer"""
        self.legal_terms = self._load_legal_terms()
        
    def _load_legal_terms(self):
        """Load legal terminology datasets"""
        return {
            'courts': [
                'Supreme Court', 'High Court', 'District Court',
                'Sessions Court', 'Magistrate Court'
            ],
            'legal_acts': [
                'Indian Penal Code', 'Code of Criminal Procedure',
                'Constitution', 'Indian Contract Act', 'Companies Act'
            ],
            'legal_terms': [
                'plaintiff', 'defendant', 'appellant', 'respondent',
                'petitioner', 'accused', 'conviction', 'acquittal'
            ]
        }
    
    def identify_entities(self, text):
        """Identify legal entities in text"""
        text_lower = text.lower()
        entities = {
            'courts': [],
            'legal_acts': [],
            'legal_terms': []
        }
        
        # Check for each type of entity
        for entity_type, terms in self.legal_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    entities[entity_type].append(term)
        
        return entities
