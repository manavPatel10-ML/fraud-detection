# Data Preprocessing Module for Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load data from a CSV file"""
        try:
            data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
            return data
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None

    def preprocess_data(self, data):
        """Preprocess the data by handling missing values and encoding categorical features"""
        # Fill missing values
        data.fillna(0, inplace=True)
        
        # One-hot encoding for categorical features
        data = pd.get_dummies(data)
        
        return data

    def split_data(self, data, target_column, test_size=0.2, random_state=42):
        """Split the dataset into training and testing sets"""
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test

# Example usage:
# preprocessor = DataPreprocessor('path_to_your_data.csv')
# data = preprocessor.load_data()
# cleaned_data = preprocessor.preprocess_data(data)
# X_train, X_test, y_train, y_test = preprocessor.split_data(cleaned_data, target_column='target')
