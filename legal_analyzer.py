# legal_analyzer.py - NLP and legal analysis functions
import re
import spacy
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LegalAnalyzer:
    def __init__(self, data_loader):
        """Initialize the Legal Analyzer with data source"""
        self.data_loader = data_loader
        
        # Load NLP model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load datasets
        self.laws_df = data_loader.load_laws_dataset()
        self.cases_df = data_loader.load_precedents_dataset()
        self.sections_df = data_loader.load_sections_dataset()
        
        # Initialize TF-IDF vectorizer for text similarity
        # Reduced max_features to avoid overfitting, added bigrams for better matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Prepare vectorized data
        self._prepare_vectorized_data()

    def _prepare_vectorized_data(self):
        """Prepare vectorized data for similarity matching"""
        # Vectorize case descriptions
        case_descriptions = self.cases_df['description'].fillna('').tolist()
        self.case_vectors = self.vectorizer.fit_transform(case_descriptions)
        
        # Create a better representation for law sections by combining title and text
        section_texts = []
        for _, row in self.sections_df.iterrows():
            # Combine law name, section number, title and text for better matching
            combined_text = f"{row['law_name']} {row['section_number']} {row['title']} {row['text']}"
            section_texts.append(combined_text)
        
        # Vectorize the combined law section texts
        self.law_vectors = self.vectorizer.transform(section_texts)

    def analyze_scenario(self, scenario_text):
        """Analyze a legal scenario and return comprehensive results"""
        # Process the text with spaCy
        doc = self.nlp(scenario_text)
        
        # Extract entities
        entities = self._extract_entities(doc)
        
        # Find relevant precedents
        precedents = self._find_relevant_precedents(scenario_text)
        
        # Identify applicable laws
        applicable_laws = self._identify_applicable_laws(scenario_text)
        
        # Generate outcome prediction
        outcome_analysis = self._predict_outcome(scenario_text, precedents, applicable_laws)
        
        # Compile results
        return {
            'entityRecognition': entities,
            'relevantPrecedents': precedents,
            'applicableLaws': applicable_laws,
            'outcomeAnalysis': outcome_analysis
        }

    def _extract_entities(self, doc):
        """Extract named entities from the text"""
        persons = []
        organizations = []
        locations = []
        dates = []
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                persons.append(ent.text)
            elif ent.label_ == 'ORG':
                organizations.append(ent.text)
            elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                locations.append(ent.text)
            elif ent.label_ == 'DATE':
                dates.append(ent.text)
        
        # Count and get unique entities
        persons = [item for item, count in Counter(persons).most_common()]
        organizations = [item for item, count in Counter(organizations).most_common()]
        locations = [item for item, count in Counter(locations).most_common()]
        dates = [item for item, count in Counter(dates).most_common()]
        
        return {
            'persons': persons,
            'organizations': organizations,
            'locations': locations,
            'dates': dates
        }

    def _find_relevant_precedents(self, scenario_text):
        """Find relevant case precedents based on scenario similarity"""
        # Vectorize input text
        scenario_vector = self.vectorizer.transform([scenario_text])
        
        # Calculate similarity with case precedents
        similarities = cosine_similarity(scenario_vector, self.case_vectors)[0]
        
        # Get top 3 most similar cases
        top_indices = similarities.argsort()[-3:][::-1]
        
        relevant_precedents = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for more matches
                case = self.cases_df.iloc[idx]
                relevant_precedents.append({
                    'case': case['name'],
                    'year': int(case['year']),
                    'relevance': 'High' if similarities[idx] > 0.4 else 'Medium',
                    'keyFindings': case['key_finding'],
                    'similarityScore': float(similarities[idx])
                })
        
        return relevant_precedents

    def _identify_applicable_laws(self, scenario_text):
        """Identify applicable laws and sections based on scenario text"""
        # Vectorize input text
        scenario_vector = self.vectorizer.transform([scenario_text])
        
        # Calculate similarity with law sections
        similarities = cosine_similarity(scenario_vector, self.law_vectors)[0]
        
        # Lower the threshold significantly to get more matches
        threshold = 0.03
        
        # Get all indices above threshold or top 5, whichever is more
        min_matches = 5
        indices_above_threshold = [i for i, sim in enumerate(similarities) if sim > threshold]
        
        if len(indices_above_threshold) < min_matches:
            # If we don't have enough matches, take top 5 regardless of threshold
            top_indices = similarities.argsort()[-min_matches:][::-1]
        else:
            # Sort the indices by similarity score
            top_indices = sorted(indices_above_threshold, key=lambda i: similarities[i], reverse=True)
        
        applicable_laws = []
        seen_laws = set()
        
        for idx in top_indices:
            section = self.sections_df.iloc[idx]
            law_name = section['law_name']
            
            # Avoid duplicate laws
            if law_name in seen_laws:
                # Find existing law and add section
                for law in applicable_laws:
                    if law['law'] == law_name:
                        if section['section_number'] not in law['sections']:
                            law['sections'].append(section['section_number'])
            
                seen_laws.add(law_name)
        
        # Always return at least one law (the most relevant) if available
        if not applicable_laws and len(self.sections_df) > 0:
            # Find the most relevant law section
            top_idx = similarities.argmax()
            section = self.sections_df.iloc[top_idx]
            applicable_laws.append({
                'law': section['law_name'],
                'sections': [section['section_number']],
                'applicability': 'Potential Match'
            })
        
        return applicable_laws

    def _predict_outcome(self, scenario_text, precedents, applicable_laws):
        """Predict case outcome based on precedents and applicable laws"""
        # This would use a trained ML model in production
        # For demo, we'll use a simple heuristic
        
        # Calculate base probability from precedent similarity
        precedent_score = 0
        if precedents:
            precedent_score = sum(p.get('similarityScore', 0) for p in precedents) / len(precedents)
        
        # Adjust based on number of applicable laws
        law_factor = min(len(applicable_laws) / 5, 1.0)
        
        # Generate probability (simplified)
        probability = 50 + (precedent_score * 30) + (law_factor * 20)
        probability = min(max(probability, 10), 90)  # Keep between 10% and 90%
        
        # Generate strategies based on probability
        strategies = []
        challenges = []
        
        if probability > 70:
            strategies.append("Proceed with litigation")
            strategies.append("Prepare comprehensive documentation")
            challenges.append("Establish burden of proof")
        elif probability > 50:
            strategies.append("Consider settlement negotiation")
            strategies.append("Gather additional evidence")
            challenges.append("Ambiguous legal interpretation")
        else:
            strategies.append("Explore alternative dispute resolution")
            strategies.append("Reassess legal position")
            challenges.append("Weak precedent support")
            challenges.append("Jurisdiction complexities")
        
        return {
            'probabilityOfSuccess': float(probability),
            'recommendedStrategy': strategies,
            'potentialChallenges': challenges
        }