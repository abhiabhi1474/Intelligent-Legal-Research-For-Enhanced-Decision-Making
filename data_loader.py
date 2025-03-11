# data_loader.py - Dataset handling functions
import os
import pandas as pd
import json

class DataLoader:
    def __init__(self, data_dir='data'):
        """Initialize the DataLoader with path to data directory"""
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize datasets
        self._init_datasets()

    def _init_datasets(self):
        """Initialize datasets if they don't exist"""
        # Check if datasets exist, otherwise create default ones
        
        # Indian Laws dataset
        if not os.path.exists(f"{self.data_dir}/indian_laws.csv"):
            self._create_default_laws_dataset()
            
        # Case Precedents dataset
        if not os.path.exists(f"{self.data_dir}/case_precedents.csv"):
            self._create_default_precedents_dataset()
            
        # Law Sections dataset
        if not os.path.exists(f"{self.data_dir}/law_sections.csv"):
            self._create_default_sections_dataset()

    def _create_default_laws_dataset(self):
        """Create default Indian laws dataset"""
        laws_data = {
            'law_id': [1, 2, 3, 4, 5],
            'law_name': [
                'Indian Penal Code',
                'Code of Criminal Procedure',
                'Indian Contract Act',
                'Hindu Marriage Act',
                'Companies Act'
            ],
            'year': [1860, 1973, 1872, 1955, 2013],
            'category': [
                'Criminal Law',
                'Criminal Procedure',
                'Civil Law',
                'Family Law',
                'Corporate Law'
            ]
        }
        
        laws_df = pd.DataFrame(laws_data)
        laws_df.to_csv(f"{self.data_dir}/indian_laws.csv", index=False)

    def _create_default_precedents_dataset(self):
        """Create default case precedents dataset"""
        precedents_data = {
            'case_id': [1, 2, 3, 4, 5],
            'name': [
                'Kesavananda Bharati v. State of Kerala',
                'Vishaka v. State of Rajasthan',
                'Olga Tellis v. Bombay Municipal Corporation',
                'M.C. Mehta v. Union of India',
                'Maneka Gandhi v. Union of India'
            ],
            'year': [1973, 1997, 1985, 1986, 1978],
            'court': [
                'Supreme Court',
                'Supreme Court',
                'Supreme Court',
                'Supreme Court',
                'Supreme Court'
            ],
            'key_finding': [
                'Basic structure doctrine of Constitution',
                'Sexual harassment at workplace guidelines',
                'Right to livelihood is part of right to life',
                'Absolute liability principle in environmental cases',
                'Right to personal liberty and due process'
            ],
            'description': [
                'Fundamental case establishing that Parliament cannot alter the basic structure of the Constitution of India.',
                'Established guidelines for dealing with sexual harassment at workplace before legislation was enacted.',
                'Case recognized the right to livelihood as an important component of right to life under Article 21.',
                'Established the principle of absolute liability in cases of hazardous industries.',
                'Expanded the interpretation of Article 21 to include right to travel abroad as part of personal liberty.'
            ]
        }
        
        precedents_df = pd.DataFrame(precedents_data)
        precedents_df.to_csv(f"{self.data_dir}/case_precedents.csv", index=False)

    def _create_default_sections_dataset(self):
        """Create default law sections dataset"""
        sections_data = {
            'section_id': range(1, 11),
            'law_name': [
                'Indian Penal Code', 'Indian Penal Code',
                'Code of Criminal Procedure', 'Code of Criminal Procedure',
                'Indian Contract Act', 'Indian Contract Act',
                'Hindu Marriage Act', 'Hindu Marriage Act',
                'Companies Act', 'Companies Act'
            ],
            'section_number': [
                'Section 302', 'Section 420',
                'Section 161', 'Section 144',
                'Section 10', 'Section 73',
                'Section 13', 'Section 5',
                'Section 149', 'Section 447'
            ],
            'title': [
                'Punishment for murder', 'Cheating and dishonesty',
                'Examination of witnesses by police', 'Power to issue order in urgent cases',
                'What agreements are contracts', 'Compensation for loss',
                'Divorce', 'Conditions for Hindu marriage',
                'Company members', 'Punishment for fraud'
            ],
            'text': [
                'Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.',
                'Whoever cheats and thereby dishonestly induces the person deceived to deliver any property shall be punished with imprisonment and fine.',
                'Police officer may examine witnesses acquainted with the facts and circumstances of the case.',
                'Magistrate may issue order in urgent cases of nuisance or apprehended danger.',
                'All agreements are contracts if they are made by free consent of parties competent to contract, for a lawful consideration and with a lawful object.',
                'When a contract has been broken, the party suffering by such breach is entitled to receive compensation for any loss caused by the breach.',
                'Any marriage may be dissolved by a decree of divorce on grounds specified in the Act.',
                'A marriage may be solemnized between any two Hindus, if the conditions specified in the Act are fulfilled.',
                'The liability of the members of a company limited by shares shall be limited to the amount unpaid on the shares.',
                'Any person guilty of fraud shall be punishable with imprisonment and fine.'
            ]
        }
        
        sections_df = pd.DataFrame(sections_data)
        sections_df.to_csv(f"{self.data_dir}/law_sections.csv", index=False)

    def load_laws_dataset(self):
        """Load the Indian laws dataset"""
        return pd.read_csv(f"{self.data_dir}/indian_laws.csv")

    def load_precedents_dataset(self):
        """Load the case precedents dataset"""
        return pd.read_csv(f"{self.data_dir}/case_precedents.csv")

    def load_sections_dataset(self):
        """Load the law sections dataset"""
        return pd.read_csv(f"{self.data_dir}/law_sections.csv")

    def get_all_laws(self):
        """Get a list of all available laws"""
        laws_df = self.load_laws_dataset()
        return laws_df[['law_id', 'law_name', 'category']].to_dict('records')

    def get_precedent_cases(self):
        """Get a list of precedent cases"""
        cases_df = self.load_precedents_dataset()
        return cases_df[['case_id', 'name', 'year', 'court']].to_dict('records')

    def add_custom_dataset(self, dataset_name, data):
        """Add a custom dataset"""
        file_path = f"{self.data_dir}/{dataset_name}.csv"
        pd.DataFrame(data).to_csv(file_path, index=False)
        return True