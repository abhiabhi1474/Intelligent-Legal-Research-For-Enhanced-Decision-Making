# app.py - Main application file
from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
from legal_analyzer import LegalAnalyzer
from data_loader import DataLoader

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize components
data_loader = DataLoader()
legal_analyzer = LegalAnalyzer(data_loader)

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_scenario():
    """API endpoint to analyze legal scenarios"""
    data = request.get_json()
    scenario_text = data.get('scenario', '')
    
    if not scenario_text:
        return jsonify({"error": "No scenario provided"}), 400
    
    # Perform the legal analysis
    analysis_result = legal_analyzer.analyze_scenario(scenario_text)
    return jsonify(analysis_result)

@app.route('/laws', methods=['GET'])
def get_laws():
    """API endpoint to retrieve available laws"""
    laws = data_loader.get_all_laws()
    return jsonify(laws)

@app.route('/precedents', methods=['GET'])
def get_precedents():
    """API endpoint to retrieve case precedents"""
    precedents = data_loader.get_precedent_cases()
    return jsonify(precedents)

if __name__ == '__main__':
    app.run(debug=True)