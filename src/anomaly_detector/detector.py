"""
Anomaly detection using Isolation Forest.
Trains on normal traffic and flags suspicious patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class AnomalyDetector:
    """
    Detects network anomalies using Isolation Forest.
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def train(self, features_df):
        """Train the anomaly detector on normal traffic."""
        print(f"Training model on {len(features_df)} samples...")
        
        X = features_df.values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        
        predictions = self.model.predict(X_scaled)
        anomalies = (predictions == -1).sum()
        
        print(f"Training complete! Found {anomalies} anomalies ({anomalies/len(X)*100:.2f}%)")
        return self.model
    
    def detect_anomalies(self, features_df):
        """Detect anomalies in new traffic."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        X = features_df.values
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        results = features_df.copy()
        results['anomaly'] = (predictions == -1).astype(int)
        results['anomaly_score'] = self.model.score_samples(X_scaled)
        
        anomaly_count = results['anomaly'].sum()
        print(f"Found {anomaly_count} anomalies ({anomaly_count/len(results)*100:.2f}%)")
        return results
    
    def save_model(self, filepath='models/anomaly_model.pkl'):
        """Save the trained model to disk."""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'trained_date': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/anomaly_model.pkl'):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
        return self


def train_anomaly_detector(csv_file, model_path='models/anomaly_model.pkl'):
    """Convenience function to train the detector on a CSV file."""
    from src.feature_extractor.extract_features import extract_features_from_file
    
    print(f"Extracting features from {csv_file}...")
    features = extract_features_from_file(csv_file)
    
    detector = AnomalyDetector()
    detector.train(features)
    detector.save_model(model_path)
    return detector


if __name__ == "__main__":
    print("Testing anomaly detector...")
    import glob
    csv_files = glob.glob('data/capture_*.csv')
    if csv_files:
        latest_file = csv_files[-1]
        print(f"Using {latest_file} for training...")
        detector = train_anomaly_detector(latest_file)
        print("\nModel training complete!")
    else:
        print("No capture files found. Run packet_capture.py first!")