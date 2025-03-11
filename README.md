# Intelligent-Legal-Research-Website
This project is a Flask-based API designed to analyze legal scenarios, retrieve relevant laws, and predict legal outcomes using Natural Language Processing (NLP) and Machine Learning (ML). It helps users by identifying applicable laws, finding past legal precedents, and estimating case success probabilities.
# Define the README content
Legal Analysis and Prediction API

This project provides a **Flask-based API** for analyzing legal scenarios, retrieving laws, and predicting legal outcomes using NLP and machine learning techniques.

## Features
- **Analyze Legal Scenarios**: Uses NLP to extract relevant legal entities and find applicable laws.
- **Retrieve Indian Laws**: Fetches legal provisions from a dataset.
- **Find Relevant Precedents**: Identifies past legal cases related to a given scenario.
- **Predict Case Outcomes**: Uses machine learning to estimate the success probability of a legal case.
# Legal Analysis and Prediction API

This project provides a **Flask-based API** for analyzing legal scenarios, retrieving laws, and predicting legal outcomes using NLP and machine learning techniques.

## Features
- **Analyze Legal Scenarios**: Uses NLP to extract relevant legal entities and find applicable laws.
- **Retrieve Indian Laws**: Fetches legal provisions from a dataset.
- **Find Relevant Precedents**: Identifies past legal cases related to a given scenario.
- **Predict Case Outcomes**: Uses machine learning to estimate the success probability of a legal case.

## Installation


###  Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```sh
pip install -r requirements.txt
```

### Download NLP Model
```sh
python -m spacy download en_core_web_sm
```

## Usage
### Run the Flask App
```sh
python app.py
```
The API will be available at `http://127.0.0.1:5000/`.

### API Endpoints
- **`GET /laws`** → Returns a list of Indian laws.
- **`GET /precedents`** → Retrieves important legal precedents.
- **`POST /analyze`** → Accepts a JSON body `{ "scenario": "legal text here" }` and returns relevant laws and cases.

## Project Structure
```
legal-analysis-api/
|-- app.py              # Main Flask application
|-- data_loader.py      # Loads law datasets
|-- legal_analyzer.py   # NLP-based legal analysis
|-- models.py           # Machine learning models
|-- requirements.txt    # Dependencies
|-- data/               # Folder containing law datasets
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

## License
MIT License

