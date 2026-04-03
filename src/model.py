# Model Training Module for Fraud Detection

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class FraudDetectionModel:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model
        
        Args:
            model_type (str): Type of model to use ('random_forest' or 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError("Model type not supported. Choose 'random_forest' or 'logistic_regression'")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"Training {self.model_type} model...")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed!")
        
        return self.model
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """
        Get probability predictions
        
        Args:
            X_test: Test features
            
        Returns:
            Probability predictions
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(filepath.replace('.pkl', '_scaler.pkl'))
        print(f"Model loaded from {filepath}")
