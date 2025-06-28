import os
import pandas as pd

class DataLoader:
    def __init__(self, data_dir='data'):
        """Initialize the DataLoader with path to data directory"""
        self.data_dir = data_dir
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

    def load_laws_dataset(self):
        """Load the Indian laws dataset"""
        laws_path = f"{self.data_dir}/indian_laws.csv"
        if not os.path.exists(laws_path):
            self.create_sample_data()
        return pd.read_csv(laws_path)

    def load_precedents_dataset(self):
        """Load the case precedents dataset"""
        precedents_path = f"{self.data_dir}/case_precedents.csv"
        if not os.path.exists(precedents_path):
            self.create_sample_data()
        return pd.read_csv(precedents_path)

    def load_sections_dataset(self):
        """Load the law sections dataset"""
        sections_path = f"{self.data_dir}/law_sections.csv"
        if not os.path.exists(sections_path):
            self.create_sample_data()
        return pd.read_csv(sections_path)

    def get_all_laws(self):
        """Get a list of all available laws"""
        return self.load_laws_dataset().to_dict('records')

    def get_precedent_cases(self):
        """Get a list of precedent cases"""
        return self.load_precedents_dataset().to_dict('records')

    def add_custom_dataset(self, dataset_name, data):
        """Add a custom dataset"""
        pd.DataFrame(data).to_csv(f"{self.data_dir}/{dataset_name}.csv", index=False)
        return True
        
    def create_sample_data(self):
        """Create sample data files if they don't exist"""
        # Sample laws data
        laws_data = [
            {'id': 1, 'name': 'Indian Penal Code', 'year': 1860, 'category': 'Criminal Law'},
            {'id': 2, 'name': 'Code of Criminal Procedure', 'year': 1973, 'category': 'Criminal Law'},
            {'id': 3, 'name': 'Indian Contract Act', 'year': 1872, 'category': 'Civil Law'},
            {'id': 4, 'name': 'Companies Act', 'year': 2013, 'category': 'Corporate Law'},
            {'id': 5, 'name': 'Constitution of India', 'year': 1950, 'category': 'Constitutional Law'}
        ]
        pd.DataFrame(laws_data).to_csv(f"{self.data_dir}/indian_laws.csv", index=False)
        
        # Sample precedents data
        precedents_data = [
            {'id': 1, 'name': 'A.K. Gopalan v. State of Madras', 'year': 1950, 'court': 'Supreme Court', 
             'key_finding': 'Article 21 only protects against arbitrary executive action, not legislative action', 
             'description': 'Constitutional validity of preventive detention', 'outcome': 'win'},
            {'id': 2, 'name': 'Kesavananda Bharati v. State of Kerala', 'year': 1973, 'court': 'Supreme Court', 
             'key_finding': 'Parliament cannot amend the basic structure of the Constitution', 
             'description': 'Constitutional validity of 24th Amendment', 'outcome': 'win'},
            {'id': 3, 'name': 'Maneka Gandhi v. Union of India', 'year': 1978, 'court': 'Supreme Court', 
             'key_finding': 'Procedure established by law must be fair, just and reasonable', 
             'description': 'Right to travel abroad', 'outcome': 'win'},
            {'id': 4, 'name': 'M.C. Mehta v. Union of India', 'year': 1986, 'court': 'Supreme Court', 
             'key_finding': 'Polluter Pays Principle', 
             'description': 'Environmental protection case', 'outcome': 'win'},
            {'id': 5, 'name': 'Mohd. Ahmed Khan v. Shah Bano Begum', 'year': 1985, 'court': 'Supreme Court', 
             'key_finding': 'Muslim women entitled to maintenance under Section 125 of CrPC', 
             'description': 'Maintenance rights of divorced Muslim women', 'outcome': 'win'}
        ]
        pd.DataFrame(precedents_data).to_csv(f"{self.data_dir}/case_precedents.csv", index=False)
        
        # Sample sections data
        sections_data = [
            {'id': 1, 'law_name': 'Indian Penal Code', 'section_number': '302', 
             'text': 'Punishment for murder - Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.'},
            {'id': 2, 'law_name': 'Indian Penal Code', 'section_number': '304B', 
             'text': 'Dowry death - Where the death of a woman is caused by any burns or bodily injury or occurs otherwise than under normal circumstances within seven years of her marriage and it is shown that soon before her death she was subjected to cruelty or harassment by her husband or any relative of her husband for, or in connection with, any demand for dowry, such death shall be called "dowry death", and such husband or relative shall be deemed to have caused her death.'},
            {'id': 3, 'law_name': 'Indian Contract Act', 'section_number': '2(h)', 
             'text': 'An agreement enforceable by law is a contract.'},
            {'id': 4, 'law_name': 'Indian Contract Act', 'section_number': '10', 
             'text': 'All agreements are contracts if they are made by the free consent of parties competent to contract, for a lawful consideration and with a lawful object, and are not hereby expressly declared to be void.'},
            {'id': 5, 'law_name': 'Constitution of India', 'section_number': 'Article 21', 
             'text': 'Protection of life and personal liberty - No person shall be deprived of his life or personal liberty except according to procedure established by law.'}
        ]
        pd.DataFrame(sections_data).to_csv(f"{self.data_dir}/law_sections.csv", index=False)
        
        print("Sample data created successfully!")
