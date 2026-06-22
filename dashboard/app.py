"""
Flask Dashboard for Network Anomaly Detector
Shows real-time alerts and visualizations
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
import glob
import joblib
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
alerts = []
captured_data = None

# ============================================
# ROUTES - Web Pages
# ============================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Alternative dashboard route"""
    return render_template('index.html')

# ============================================
# API ROUTES - Data Endpoints
# ============================================

@app.route('/api/status')
def get_status():
    """Get system status"""
    model_loaded = model is not None
    
    # Get latest capture file
    csv_files = glob.glob('data/capture_*.csv')
    latest_file = csv_files[-1] if csv_files else None
    
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'alerts_count': len(alerts),
        'capture_file': latest_file,
        'last_update': datetime.now().isoformat()
    })

@app.route('/api/capture', methods=['POST'])
def capture_packets():
    """Capture new packets"""
    global captured_data
    
    try:
        data = request.json
        count = data.get('count', 50)
        
        # Import and run packet capture
        from src.packet_capture import quick_capture
        
        filename = quick_capture(count=count)
        
        if filename:
            return jsonify({
                'success': True,
                'message': f'Captured packets saved to {filename}',
                'file': filename
            })
        else:
            return jsonify({'error': 'No packets captured'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the anomaly detection model"""
    global model, scaler
    
    try:
        # Find latest capture file
        csv_files = glob.glob('data/capture_*.csv')
        if not csv_files:
            return jsonify({'error': 'No capture files found'}), 400
        
        latest_file = csv_files[-1]
        
        # Import and train
        from train_model import extract_features_from_csv, train_model
        
        # Train the model
        model = train_model(latest_file)
        
        # Load the saved model
        model_data = joblib.load('models/anomaly_model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_anomalies():
    """Detect anomalies in the latest capture"""
    global model, scaler, alerts
    
    try:
        if model is None:
            return jsonify({'error': 'Model not trained yet!'}), 400
        
        # Find latest capture file
        csv_files = glob.glob('data/capture_*.csv')
        if not csv_files:
            return jsonify({'error': 'No capture files found'}), 400
        
        latest_file = csv_files[-1]
        
        # Load and extract features
        from train_model import extract_features_from_csv
        features = extract_features_from_csv(latest_file)
        
        # Scale features
        X_scaled = scaler.transform(features.values)
        
        # Predict anomalies
        predictions = model.predict(X_scaled)
        anomaly_scores = model.score_samples(X_scaled)
        
        # Create results
        results = features.copy()
        results['anomaly'] = (predictions == -1).astype(int)
        results['anomaly_score'] = anomaly_scores
        
        # Count anomalies - CONVERT TO INT
        anomaly_count = int(results['anomaly'].sum())
        anomaly_indices = results[results['anomaly'] == 1].index.tolist()
        
        # Load original data for context
        original_data = pd.read_csv(latest_file)
        
        # Create alerts for anomalies
        new_alerts = []
        for idx in anomaly_indices:
            if idx < len(original_data):
                packet = original_data.iloc[idx]
                
                # Convert all numpy types to Python types
                src_port_val = packet.get('src_port', 0)
                dst_port_val = packet.get('dst_port', 0)
                length_val = packet.get('length', 0)
                anomaly_score_val = results.loc[idx, 'anomaly_score']
                
                # Handle NaN values
                if pd.isna(src_port_val):
                    src_port_val = 0
                if pd.isna(dst_port_val):
                    dst_port_val = 0
                if pd.isna(length_val):
                    length_val = 0
                
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'src_ip': str(packet.get('src_ip', 'Unknown')),
                    'dst_ip': str(packet.get('dst_ip', 'Unknown')),
                    'src_port': int(src_port_val),
                    'dst_port': int(dst_port_val),
                    'protocol': str(packet.get('flags', 'Unknown')),
                    'packet_size': int(length_val),
                    'anomaly_score': float(anomaly_score_val),
                    'severity': 'High' if float(anomaly_score_val) < -0.5 else 'Medium'
                }
                new_alerts.append(alert)
                alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(alerts) > 100:
            alerts = alerts[-100:]
        
        # Prepare summary - CONVERT ALL TO PYTHON TYPES
        summary = {
            'total_packets': int(len(results)),
            'anomalies': int(anomaly_count),
            'anomaly_percentage': float(round(anomaly_count / len(results) * 100, 2)),
            'recent_alerts': new_alerts[-10:] if new_alerts else []
        }
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"Error in detect_anomalies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify({
        'alerts': alerts[-20:],
        'total': len(alerts)
    })

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    global alerts
    alerts = []
    return jsonify({'success': True, 'message': 'Alerts cleared'})

# ============================================
# MAIN - Run the App
# ============================================

if __name__ == '__main__':
    # Try to load existing model
    try:
        if os.path.exists('models/anomaly_model.pkl'):
            model_data = joblib.load('models/anomaly_model.pkl')
            model = model_data['model']
            scaler = model_data['scaler']
            print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ No model found. Train the model first.")
    
    print("\n" + "="*50)
    print("🚀 NETWORK ANOMALY DETECTOR DASHBOARD")
    print("="*50)
    print("📍 URL: http://127.0.0.1:5000")
    print("📍 Dashboard: http://127.0.0.1:5000/dashboard")
    print("="*50)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, port=5000)