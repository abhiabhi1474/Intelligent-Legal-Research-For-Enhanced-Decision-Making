import re
import pandas as pd
import numpy as np
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

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
        self.classification_report = classification_report(y_test, y_pred)
        
        print(f"Model Accuracy: {self.model_accuracy:.2%}")
        print("Classification Report:")
        print(self.classification_report)

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

    def extract_semantic_meaning(self, scenario_text):
        """Extract semantic meaning and key entities from scenario"""
        if not self.use_spacy:
            return {
                'entities': [],
                'noun_chunks': [],
                'lemmatized_text': scenario_text
            }

        # Process the scenario text with spaCy
        doc = self.nlp(scenario_text)

        # Extract key semantic information
        semantic_analysis = {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
            'lemmatized_text': ' '.join([token.lemma_ for token in doc]),
            'pos_tags': [(token.text, token.pos_) for token in doc],
            'dependencies': [(token.text, token.dep_) for token in doc]
        }

        return semantic_analysis

    def find_synonyms_and_semantically_similar_terms(self, scenario_text):
        """Find synonyms and semantically similar terms"""
        if not self.use_spacy:
            return {'synonyms': [], 'similar_terms': []}

        # Process the scenario text
        doc = self.nlp(scenario_text)

        # Find synonyms and similar terms
        similar_terms = {}
        for token in doc:
            # Skip stop words and punctuation
            if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                try:
                    # Find similar words based on word vectors
                    similar_words = [w for w in token.vocab if w.is_lower and w.has_vector and w.is_oov == token.is_oov]
                    similar_words = sorted(similar_words, key=lambda w: token.similarity(w), reverse=True)[:3]
                    
                    similar_terms[token.text] = [w.text for w in similar_words]
                except:
                    similar_terms[token.text] = []

        return {
            'lemmatized_text': ' '.join([token.lemma_ for token in doc]),
            'semantically_similar_terms': similar_terms
        }

    def analyze_scenario(self, scenario_text):
        """Enhanced scenario analysis with NLP-powered insights"""
        # Extract semantic meaning
        semantic_meaning = self.extract_semantic_meaning(scenario_text)
        
        # Find synonyms and similar terms
        semantic_similarity = self.find_synonyms_and_semantically_similar_terms(scenario_text)
        
        # Existing analysis methods
        precedents = self._find_relevant_precedents(scenario_text)
        applicable_laws = self._identify_applicable_laws(scenario_text)
        outcome_analysis = self._predict_outcome(scenario_text)
        
        return {
            'semanticMeaning': semantic_meaning,
            'semanticSimilarity': semantic_similarity,
            'relevantPrecedents': precedents,
            'applicableLaws': applicable_laws,
            'outcomeAnalysis': outcome_analysis
        }
