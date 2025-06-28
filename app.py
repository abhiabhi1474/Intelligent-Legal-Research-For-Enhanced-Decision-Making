from flask import Flask, render_template, request, jsonify
import pandas as pd
import nltk
import os
from legal_analyzer import LegalAnalyzer
from data_loader import DataLoader

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.abspath('templates'),
            static_folder=os.path.abspath('static'))

# Ensure the templates directory exists
os.makedirs('templates', exist_ok=True)

# Copy index.html to templates directory if it doesn't exist
if not os.path.exists('templates/index.html'):
    with open('index.html', 'r') as f:
        content = f.read()
    with open('templates/index.html', 'w') as f:
        f.write(content)

# Download necessary NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    print("Warning: NLTK data download failed. The app may not function properly.")

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
    
    # Add entity recognition
   
    
    return jsonify(analysis_result)

@app.route('/laws', methods=['GET'])
def get_laws():
    """API endpoint to retrieve available laws"""
    laws = data_loader.load_laws_dataset().to_dict(orient='records')
    return jsonify(laws)

@app.route('/precedents', methods=['GET'])
def get_precedents():
    """API endpoint to retrieve case precedents"""
    precedents = data_loader.load_precedents_dataset().to_dict(orient='records')
    return jsonify(precedents)

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Check if required CSV files exist, if not create sample data
    if not os.path.exists('data/indian_laws.csv'):
        data_loader.create_sample_data()
        
    app.run(debug=True)
