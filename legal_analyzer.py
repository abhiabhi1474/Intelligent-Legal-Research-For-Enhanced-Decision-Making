import pandas as pd
import numpy as np
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class LegalAnalyzer:
    def __init__(self, data_loader):
        """Initialize the Legal Analyzer with data source and NLP processing"""
        self.data_loader = data_loader
        
        # Load spaCy models
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
        except:
            print("Spacy model not found. Basic NLP features will be limited.")
            self.use_spacy = False
        
        # Load Sentence Transformer model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load datasets
        self.laws_df = data_loader.load_laws_dataset()
        self.cases_df = data_loader.load_precedents_dataset()
        self.sections_df = data_loader.load_sections_dataset()
        
        # Prepare embedded data
        self._prepare_embedded_data()
        
        # Train ML model
        self._train_model()

    def _prepare_embedded_data(self):
        """Prepare sentence embeddings for similarity matching"""
        # Embed case descriptions
        case_descriptions = self.cases_df['description'].fillna('').tolist()
        self.case_embeddings = self.embedding_model.encode(case_descriptions, convert_to_tensor=False)
        
        # Embed law sections
        law_sections = self.sections_df['text'].fillna('').tolist()
        self.law_embeddings = self.embedding_model.encode(law_sections, convert_to_tensor=False)

    def _train_model(self):
        """Train a logistic regression model for outcome prediction"""
        features = self.case_embeddings
        labels = [1 if outcome == 'win' else 0 for outcome in self.cases_df['outcome']]
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.model_accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {self.model_accuracy:.2%}")
        
    def _calculate_similarity(self, query_embedding, corpus_embeddings):
        """Calculate similarity between query and corpus using dot product"""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        corpus_norms = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        # Calculate dot product similarities
        similarities = np.dot(corpus_norms, query_norm)
        return similarities

    def _find_relevant_precedents(self, scenario_text):
        """Find relevant case precedents based on scenario similarity"""
        # Embed input scenario
        scenario_embedding = self.embedding_model.encode([scenario_text], convert_to_tensor=False)[0]
        
        # Calculate similarity with case precedents
        similarities = self._calculate_similarity(scenario_embedding, self.case_embeddings)
        
        # Get top 3 most similar cases
        top_indices = similarities.argsort()[-3:][::-1]
        
        relevant_precedents = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                case = self.cases_df.iloc[idx]
                relevant_precedents.append({
                    'case': case['name'],
                    'year': int(case['year']),
                    'relevance': 'High' if similarities[idx] > 0.5 else 'Medium',
                    'keyFindings': case['key_finding'],
                    'similarityScore': float(similarities[idx])
                })
        
        return relevant_precedents

    def _identify_applicable_laws(self, scenario_text):
        """Identify applicable laws and sections based on scenario text"""
        # Embed input scenario
        scenario_embedding = self.embedding_model.encode([scenario_text], convert_to_tensor=False)[0]
        
        # Calculate similarity with law sections
        similarities = self._calculate_similarity(scenario_embedding, self.law_embeddings)
        
        # Get top 5 most similar sections
        top_indices = similarities.argsort()[-5:][::-1]
        
        applicable_laws = []
        seen_laws = set()
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                section = self.sections_df.iloc[idx]
                law_name = section['law_name']
                
                # Avoid duplicate laws
                if law_name in seen_laws:
                    # Find existing law and add section
                    for law in applicable_laws:
                        if law['law'] == law_name:
                            if section['section_number'] not in law['sections']:
                                law['sections'].append(section['section_number'])
                else:
                    # Add new law with section
                    applicable_laws.append({
                        'law': law_name,
                        'sections': [section['section_number']],
                        'applicability': 'Direct' if similarities[idx] > 0.5 else 'Indirect'
                    })
                    seen_laws.add(law_name)
        
        return applicable_laws

    def _predict_outcome(self, scenario_text):
        """Predict case outcome using trained model"""
        # Embed scenario
        scenario_embedding = self.embedding_model.encode([scenario_text], convert_to_tensor=False)
        scenario_embedding = self.scaler.transform(scenario_embedding)
        
        # Predict probability
        probability = self.model.predict_proba(scenario_embedding)[0][1] * 100
        
        # Generate appropriate recommendations based on probability
        if probability > 70:
            recommendations = ['Proceed with litigation', 'Gather additional supporting evidence', 'Prepare for possible settlement offers']
            challenges = ['Potential delays in court proceedings', 'Possibility of appeal by opposing party']
        elif probability > 50:
            recommendations = ['Proceed with litigation', 'Consider settlement negotiations', 'Strengthen case with expert testimony']
            challenges = ['Moderate case complexity', 'Some precedent inconsistencies']
        elif probability > 30:
            recommendations = ['Explore settlement options', 'Strengthen weak areas of the case', 'Consider alternative dispute resolution']
            challenges = ['Insufficient precedent support', 'Legal ambiguities in similar cases']
        else:
            recommendations = ['Consider settlement', 'Evaluate alternative legal strategies', 'Reassess case merits']
            challenges = ['Weak legal foundation', 'Strong opposing precedents', 'High risk of adverse outcome']
        
        return {
            'probabilityOfSuccess': float(probability),
            'recommendedStrategy': recommendations,
            'potentialChallenges': challenges
        }
    def analyze_scenario(self, scenario_text):
        
        precedents = self._find_relevant_precedents(scenario_text)
        applicable_laws = self._identify_applicable_laws(scenario_text)
        outcome_analysis = self._predict_outcome(scenario_text)
        return {
            'relevantPrecedents': precedents,
            'applicableLaws': applicable_laws,
            'outcomeAnalysis': outcome_analysis
        }
