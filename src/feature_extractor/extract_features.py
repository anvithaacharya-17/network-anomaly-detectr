"""
Extracts features from network packets for ML analysis.
Converts raw packets into numbers the AI can understand.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class FeatureExtractor:
    """
    Extracts features from packet data.
    """
    
    def __init__(self):
        self.features = []
        
    def extract_from_csv(self, csv_file):
        """
        Extract features from a CSV file of packets.
        """
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} packets from {csv_file}")
        features_df = self.extract_features(df)
        return features_df
    
    def extract_features(self, df):
        """
        Extract features from a DataFrame of packets.
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Calculate inter-arrival time
        df['time_diff'] = df['timestamp'].diff()
        df['time_diff'] = df['time_diff'].fillna(0)
        
        # Protocol names
        protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
        df['protocol_name'] = df['protocol'].map(protocol_map).fillna('OTHER')
        
        # Common ports
        common_ports = [80, 443, 53, 22, 25, 21, 20, 110, 143, 993, 995]
        df['is_common_port'] = df['dst_port'].apply(lambda x: 1 if x in common_ports else 0)
        
        # Port zero (scanning behavior)
        df['port_zero'] = df['dst_port'].apply(lambda x: 1 if x == 0 else 0)
        
        # Packet size features
        df['is_small_packet'] = (df['length'] < 64).astype(int)
        df['is_large_packet'] = (df['length'] > 1500).astype(int)
        
        # Protocol flags
        df['protocol_tcp'] = (df['protocol_name'] == 'TCP').astype(int)
        df['protocol_udp'] = (df['protocol_name'] == 'UDP').astype(int)
        df['protocol_icmp'] = (df['protocol_name'] == 'ICMP').astype(int)
        
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Flow features
        src_counts = df.groupby('src_ip').size().reset_index(name='src_packet_count')
        df = df.merge(src_counts, on='src_ip', how='left')
        
        dst_counts = df.groupby('dst_ip').size().reset_index(name='dst_packet_count')
        df = df.merge(dst_counts, on='dst_ip', how='left')
        
        # Select features for ML
        feature_columns = [
            'length', 'ttl', 'time_diff', 'is_common_port', 'port_zero',
            'is_small_packet', 'is_large_packet', 'protocol_tcp', 
            'protocol_udp', 'protocol_icmp', 'hour', 
            'src_packet_count', 'dst_packet_count'
        ]
        
        available_features = [col for col in feature_columns if col in df.columns]
        features_df = df[available_features]
        features_df = features_df.fillna(0)
        
        print(f"Extracted {len(features_df)} samples with {len(available_features)} features")
        return features_df


def extract_features_from_file(csv_file):
    """Convenience function to extract features from a CSV file."""
    extractor = FeatureExtractor()
    return extractor.extract_from_csv(csv_file)


if __name__ == "__main__":
    # Test the feature extractor
    print("Testing feature extraction...")
    import glob
    csv_files = glob.glob('data/capture_*.csv')
    if csv_files:
        latest_file = csv_files[-1]
        print(f"Found capture file: {latest_file}")
        features = extract_features_from_file(latest_file)
        print(f"Extracted features shape: {features.shape}")
        print("\nFeature preview:")
        print(features.head())
    else:
        print("No capture files found. Run packet_capture.py first!")