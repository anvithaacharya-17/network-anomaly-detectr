"""
Standalone script to train the anomaly detector.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob
from datetime import datetime

def extract_features_from_csv(csv_file):
    """Extract features from CSV file."""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} packets from {csv_file}")
    
    # Create features
    df['time_diff'] = df['timestamp'].diff().fillna(0)
    
    # Protocol mapping
    protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
    df['protocol_name'] = df['protocol'].map(protocol_map).fillna('OTHER')
    
    # Common ports
    common_ports = [80, 443, 53, 22, 25, 21, 20, 110, 143, 993, 995]
    df['is_common_port'] = df['dst_port'].apply(lambda x: 1 if x in common_ports else 0)
    df['port_zero'] = df['dst_port'].apply(lambda x: 1 if x == 0 else 0)
    df['is_small_packet'] = (df['length'] < 64).astype(int)
    df['is_large_packet'] = (df['length'] > 1500).astype(int)
    
    # Protocol flags
    df['protocol_tcp'] = (df['protocol_name'] == 'TCP').astype(int)
    df['protocol_udp'] = (df['protocol_name'] == 'UDP').astype(int)
    df['protocol_icmp'] = (df['protocol_name'] == 'ICMP').astype(int)
    
    # Time features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    
    # Flow features
    src_counts = df.groupby('src_ip').size().reset_index(name='src_packet_count')
    df = df.merge(src_counts, on='src_ip', how='left')
    dst_counts = df.groupby('dst_ip').size().reset_index(name='dst_packet_count')
    df = df.merge(dst_counts, on='dst_ip', how='left')
    
    # Select features
    feature_columns = [
        'length', 'ttl', 'time_diff', 'is_common_port', 'port_zero',
        'is_small_packet', 'is_large_packet', 'protocol_tcp', 
        'protocol_udp', 'protocol_icmp', 'hour',
        'src_packet_count', 'dst_packet_count'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    features_df = df[available_features].fillna(0)
    
    print(f"Extracted {len(features_df)} samples with {len(available_features)} features")
    return features_df


def train_model(csv_file):
    """Train the anomaly detection model."""
    print(f"\nTraining model on {csv_file}...")
    
    # Extract features
    features = extract_features_from_csv(csv_file)
    X = features.values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    model.fit(X_scaled)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'scaler': scaler,
        'trained_date': datetime.now().isoformat()
    }
    joblib.dump(model_data, 'models/anomaly_model.pkl')
    
    # Show results
    predictions = model.predict(X_scaled)
    anomalies = (predictions == -1).sum()
    
    print(f"Training complete!")
    print(f"Found {anomalies} anomalies ({anomalies/len(X)*100:.2f}%)")
    print(f"Model saved to models/anomaly_model.pkl")
    
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("TRAINING ANOMALY DETECTOR")
    print("=" * 50)
    
    csv_files = glob.glob('data/capture_*.csv')
    if csv_files:
        latest_file = csv_files[-1]
        print(f"Using: {latest_file}\n")
        train_model(latest_file)
    else:
        print("No capture files found!")