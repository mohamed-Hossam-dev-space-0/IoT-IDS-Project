#!/usr/bin/env python3
"""
AI-Based Intrusion Detection System for IoT Networks - SIMPLIFIED VERSION
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def analyze_mode():
    """Analyze IoT architecture and attacks"""
    print("\n" + "="*60)
    print("IoT ARCHITECTURE & ATTACK ANALYSIS")
    print("="*60)
    
    print("\n📊 IoT System Architecture:")
    print("-"*40)
    print("""
    1. PERCEPTION LAYER (Sensors/Devices):
       • Temperature sensors
       • Motion detectors  
       • Cameras
       • Smart actuators
    
    2. NETWORK LAYER (Communication):
       • Wi-Fi routers
       • Zigbee/Z-Wave gateways
       • Bluetooth modules
       • 5G/LTE connections
    
    3. APPLICATION LAYER (Services):
       • Cloud platforms
       • Mobile applications
       • Data analytics
       • User interfaces
    """)
    
    print("\n⚠️  Common IoT Attacks:")
    print("-"*40)
    print("""
    1. DENIAL OF SERVICE (DoS):
       • Mechanism: Flood devices with traffic
       • Target: Network bandwidth, device resources
       • Detection: High packet rate, small packets
    
    2. MAN-IN-THE-MIDDLE (MITM):
       • Mechanism: Intercept communications
       • Target: Data confidentiality, integrity
       • Detection: Protocol anomalies, timing issues
    
    3. DATA INJECTION:
       • Mechanism: Send malicious data
       • Target: System decisions, data integrity
       • Detection: Anomalous data patterns
    """)

def demo_mode():
    """Run complete IDS demonstration"""
    print("\n" + "="*60)
    print("AI-BASED IDS DEMONSTRATION")
    print("="*60)
    
    # 1. Generate IoT data
    print("\n[1] Generating IoT Network Data...")
    np.random.seed(42)
    n_samples = 5000
    
    # Create features
    data = {
        'packet_size': np.random.normal(500, 150, n_samples),
        'packet_count': np.random.poisson(5, n_samples),
        'error_rate': np.random.uniform(0, 0.1, n_samples),
        'duration': np.random.exponential(1.0, n_samples),
        'protocol': np.random.choice([0, 1, 2], n_samples)  # 0=TCP, 1=UDP, 2=ICMP
    }
    
    df = pd.DataFrame(data)
    y = np.zeros(n_samples)
    
    # Create attacks (20% of data)
    n_attacks = int(0.2 * n_samples)
    attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
    y[attack_indices] = 1
    
    # Make attack samples different
    for idx in attack_indices:
        if idx % 2 == 0:  # DoS attacks
            df.loc[idx, 'packet_count'] *= np.random.randint(10, 100)
            df.loc[idx, 'error_rate'] = np.random.uniform(0.5, 1.0)
        else:  # MITM attacks
            df.loc[idx, 'packet_size'] *= np.random.uniform(1.5, 3.0)
            df.loc[idx, 'protocol'] = np.random.choice([0, 1, 2], p=[0.1, 0.1, 0.8])
    
    X = df.values
    
    print(f"✅ Generated {n_samples} IoT traffic samples")
    print(f"   • Normal traffic: {sum(y==0)} samples")
    print(f"   • Attack traffic: {sum(y==1)} samples (DoS & MITM)")
    
    # 2. Split data
    print("\n[2] Preparing Data for AI Training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   • Training set: {X_train.shape[0]} samples")
    print(f"   • Test set: {X_test.shape[0]} samples")
    
    # 3. Train AI model
    print("\n[3] Training AI Intrusion Detection Model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✅ AI model trained successfully!")
    
    # 4. Evaluate
    print("\n[4] Evaluating IDS Performance...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Detection results
    tp = sum((y_test == 1) & (y_pred == 1))  # True attacks detected
    fp = sum((y_test == 0) & (y_pred == 1))  # False alarms
    fn = sum((y_test == 1) & (y_pred == 0))  # Missed attacks
    
    print("\n📊 PERFORMANCE METRICS:")
    print("-"*30)
    print(f"   Accuracy:    {accuracy*100:.2f}%")
    print(f"   Precision:   {precision*100:.2f}%")
    print(f"   Recall:      {recall*100:.2f}%")
    print(f"   F1-Score:    {f1*100:.2f}%")
    
    print("\n🎯 DETECTION RESULTS:")
    print("-"*30)
    print(f"   Attacks detected: {tp} / {sum(y_test==1)}")
    print(f"   False alarms:     {fp}")
    print(f"   Missed attacks:   {fn}")
    
    # 5. Show sample detections
    print("\n[5] Sample Attack Detections:")
    print("-"*40)
    
    # Find some attack and normal samples
    attack_samples = []
    normal_samples = []
    
    for i in range(len(y_test)):
        if y_test[i] == 1 and len(attack_samples) < 3:
            attack_samples.append(i)
        elif y_test[i] == 0 and len(normal_samples) < 2:
            normal_samples.append(i)
        if len(attack_samples) >= 3 and len(normal_samples) >= 2:
            break
    
    print("\n🔴 ATTACK SAMPLES (Should be detected):")
    for idx in attack_samples:
        pred = "🚨 ATTACK" if y_pred[idx] == 1 else "✅ Normal"
        actual = "Attack"
        prob = y_proba[idx] * 100
        print(f"   Sample {idx}: AI predicted = {pred} ({prob:.1f}% confidence)")
        print(f"        Features: [Packet size: {X_test[idx][0]:.0f}, Count: {X_test[idx][1]:.0f}]")
    
    print("\n🟢 NORMAL SAMPLES (Should be allowed):")
    for idx in normal_samples:
        pred = "🚨 ATTACK" if y_pred[idx] == 1 else "✅ Normal"
        actual = "Normal"
        prob = (1 - y_proba[idx]) * 100
        print(f"   Sample {idx}: AI predicted = {pred} ({prob:.1f}% confidence)")
        print(f"        Features: [Packet size: {X_test[idx][0]:.0f}, Count: {X_test[idx][1]:.0f}]")
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nYour AI-based IDS is successfully detecting IoT attacks!")
    print("Use these results for your project report and presentation.")

def main():
    parser = argparse.ArgumentParser(description='AI-Based IDS for IoT Networks')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['analyze', 'demo'],
                       help='Mode: analyze (architecture) or demo (IDS detection)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AI-Based Intrusion Detection System for IoT Networks")
    print("="*60)
    
    if args.mode == 'analyze':
        analyze_mode()
    else:
        demo_mode()

if __name__ == "__main__":
    main()