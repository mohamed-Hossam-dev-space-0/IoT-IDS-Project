import numpy as np
import pandas as pd
import random
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AttackSimulator:
    def __init__(self):
        self.attack_patterns = self._define_attack_patterns()
        self.iot_device_profiles = self._define_iot_device_profiles()
        self.current_attack = None
    
    def _define_attack_patterns(self):
        """Define detailed attack patterns"""
        return {
            'dos': {
                'name': 'Denial of Service',
                'subtypes': ['UDP Flood', 'TCP SYN Flood', 'HTTP Flood', 'ICMP Flood'],
                'indicators': ['High packet rate', 'Small packet size', 'Spoofed source IPs'],
                'target_ports': [80, 443, 53, 123],
                'duration_range': (10, 300),
                'intensity': 'High'
            },
            'mitm': {
                'name': 'Man-in-the-Middle',
                'subtypes': ['ARP Spoofing', 'DNS Spoofing', 'SSL Stripping', 'Session Hijacking'],
                'indicators': ['MAC address anomalies', 'Duplicate IPs', 'Certificate errors'],
                'target_protocols': ['HTTP', 'HTTPS', 'MQTT', 'CoAP'],
                'duration_range': (30, 600),
                'intensity': 'Medium'
            },
            'eavesdropping': {
                'name': 'Eavesdropping',
                'subtypes': ['Packet Sniffing', 'Traffic Analysis', 'Side-channel Attack'],
                'indicators': ['Promiscuous mode detection', 'Unusual read operations', 'Timing anomalies'],
                'target_data': ['Credentials', 'Sensitive data', 'Encryption keys'],
                'duration_range': (60, 1800),
                'intensity': 'Low'
            },
            'data_injection': {
                'name': 'Data Injection',
                'subtypes': ['SQL Injection', 'Command Injection', 'Buffer Overflow', 'XML Injection'],
                'indicators': ['Malformed packets', 'Oversized payloads', 'Protocol violations'],
                'target_services': ['Web servers', 'Databases', 'APIs', 'Control systems'],
                'duration_range': (5, 120),
                'intensity': 'High'
            },
            'replay_attack': {
                'name': 'Replay Attack',
                'subtypes': ['Session Replay', 'Command Replay', 'Authentication Replay'],
                'indicators': ['Duplicate packets', 'Sequence number issues', 'Timestamp anomalies'],
                'target_actions': ['Authentication', 'Control commands', 'Financial transactions'],
                'duration_range': (1, 60),
                'intensity': 'Medium'
            }
        }
    
    def _define_iot_device_profiles(self):
        """Define profiles for different IoT devices"""
        return {
            'smart_camera': {
                'protocols': ['RTSP', 'HTTP', 'HTTPS'],
                'ports': [554, 80, 443, 8000],
                'packet_size_range': (1000, 5000),
                'frequency_range': (0.033, 0.1),  # 30-100 FPS
                'data_type': 'Video stream',
                'sensitivity': 'High'
            },
            'smart_thermostat': {
                'protocols': ['MQTT', 'HTTP'],
                'ports': [1883, 8883, 80],
                'packet_size_range': (100, 500),
                'frequency_range': (60, 300),  # Every 1-5 minutes
                'data_type': 'Sensor readings',
                'sensitivity': 'Medium'
            },
            'smart_lock': {
                'protocols': ['Zigbee', 'Bluetooth', 'Wi-Fi'],
                'ports': [17756, 8080],
                'packet_size_range': (50, 200),
                'frequency_range': (3600, 86400),  # Very infrequent
                'data_type': 'Access control',
                'sensitivity': 'Critical'
            },
            'motion_sensor': {
                'protocols': ['Z-Wave', 'Wi-Fi'],
                'ports': [4123, 80],
                'packet_size_range': (20, 100),
                'frequency_range': (1, 10),  # When motion detected
                'data_type': 'Binary events',
                'sensitivity': 'Medium'
            },
            'voice_assistant': {
                'protocols': ['HTTP', 'WebSocket', 'HTTPS'],
                'ports': [80, 443, 9000],
                'packet_size_range': (500, 2000),
                'frequency_range': (0.5, 5),  # During interaction
                'data_type': 'Voice commands',
                'sensitivity': 'High'
            }
        }
    
    def generate_live_traffic(self, batch_size=100, attack_probability=0.2):
        """
        Generate a batch of live network traffic with potential attacks
        
        Args:
            batch_size: Number of samples to generate
            attack_probability: Probability of generating an attack sample
        
        Returns:
            Dictionary with features, labels, and attack info
        """
        features = []
        labels = []
        attack_info = []
        
        for i in range(batch_size):
            # Decide if this is an attack
            is_attack = np.random.random() < attack_probability
            
            if is_attack:
                # Generate attack traffic
                attack_type = np.random.choice(list(self.attack_patterns.keys()))
                subtype = np.random.choice(self.attack_patterns[attack_type]['subtypes'])
                
                # Generate features based on attack type
                if attack_type == 'dos':
                    feat = self._generate_dos_features()
                elif attack_type == 'mitm':
                    feat = self._generate_mitm_features()
                elif attack_type == 'eavesdropping':
                    feat = self._generate_eavesdropping_features()
                elif attack_type == 'data_injection':
                    feat = self._generate_data_injection_features()
                else:  # replay_attack
                    feat = self._generate_replay_attack_features()
                
                features.append(feat)
                labels.append(1)
                attack_info.append({
                    'type': attack_type,
                    'subtype': subtype,
                    'timestamp': datetime.now().isoformat(),
                    'intensity': self.attack_patterns[attack_type]['intensity']
                })
                
            else:
                # Generate normal traffic
                device_type = np.random.choice(list(self.iot_device_profiles.keys()))
                feat = self._generate_normal_features(device_type)
                
                features.append(feat)
                labels.append(0)
                attack_info.append({
                    'type': 'normal',
                    'subtype': device_type,
                    'timestamp': datetime.now().isoformat(),
                    'intensity': 'None'
                })
        
        return {
            'features': np.array(features),
            'labels': np.array(labels),
            'attack_info': attack_info,
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'attack_count': sum(labels)
        }
    
    def _generate_normal_features(self, device_type):
        """Generate normal traffic features for a specific IoT device"""
        profile = self.iot_device_profiles[device_type]
        
        # Base features (24 features as in data_loader_enhanced.py)
        features = np.zeros(24)
        
        # Packet size (feature 0)
        features[0] = np.random.uniform(*profile['packet_size_range'])
        
        # Packet count (feature 1)
        if device_type == 'smart_camera':
            features[1] = np.random.poisson(30)  # High for video
        elif device_type == 'motion_sensor':
            features[1] = np.random.poisson(1)   # Low, event-based
        else:
            features[1] = np.random.poisson(5)   # Medium
        
        # Flow duration (feature 2)
        if device_type in ['smart_camera', 'voice_assistant']:
            features[2] = np.random.exponential(30)  # Longer sessions
        else:
            features[2] = np.random.exponential(5)   # Shorter sessions
        
        # Source bytes (feature 3)
        features[3] = np.random.lognormal(6, 1)
        
        # Destination bytes (feature 4)
        features[4] = np.random.lognormal(6, 1)
        
        # Protocol (feature 5) - encoded
        protocol_mapping = {'RTSP': 0, 'HTTP': 1, 'HTTPS': 2, 'MQTT': 3, 
                           'Zigbee': 4, 'Bluetooth': 5, 'Z-Wave': 6, 'WebSocket': 7}
        protocol = np.random.choice(profile['protocols'])
        features[5] = protocol_mapping.get(protocol, 0)
        
        # Service (feature 6) - encoded
        service_mapping = {'video': 0, 'sensor': 1, 'control': 2, 'voice': 3}
        features[6] = service_mapping.get(profile['data_type'].split()[0].lower(), 0)
        
        # Flag (feature 7)
        flags = ['SF', 'S0', 'S1', 'S2', 'S3', 'OTH']
        flag_encoding = [0, 1, 2, 3, 4, 5]
        features[7] = np.random.choice(flag_encoding, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.1])
        
        # Statistical features (8-23)
        features[8] = np.random.poisson(5)   # src_count
        features[9] = np.random.poisson(5)   # dst_count
        features[10] = np.random.beta(8, 2)  # same_srv_rate
        features[11] = np.random.beta(2, 8)  # diff_srv_rate
        features[12] = np.random.beta(3, 7)  # srv_diff_host_rate
        features[13] = np.random.poisson(10) # dst_host_count
        features[14] = np.random.poisson(6)  # dst_host_srv_count
        features[15] = np.random.beta(7, 3)  # dst_host_same_srv_rate
        features[16] = np.random.beta(3, 7)  # dst_host_diff_srv_rate
        features[17] = np.random.beta(6, 4)  # dst_host_same_src_port_rate
        features[18] = np.random.beta(4, 6)  # dst_host_srv_diff_host_rate
        features[19] = np.random.beta(1, 9)  # dst_host_serror_rate
        features[20] = np.random.beta(1, 9)  # dst_host_srv_serror_rate
        features[21] = np.random.beta(1, 9)  # dst_host_rerror_rate
        features[22] = np.random.beta(1, 9)  # dst_host_srv_rerror_rate
        features[23] = np.random.uniform(0, 8)  # entropy
        
        return features
    
    def _generate_dos_features(self):
        """Generate features for DoS attack"""
        features = self._generate_normal_features(np.random.choice(list(self.iot_device_profiles.keys())))
        
        # Modify for DoS characteristics
        features[0] = np.random.randint(64, 128)  # Small packets
        features[1] *= np.random.randint(50, 500)  # Very high packet count
        features[2] = np.random.exponential(0.1)   # Very short duration
        features[8] *= 10  # High src_count
        features[13] *= 20  # Very high dst_host_count
        features[19] = np.random.beta(8, 2)  # High error rate
        features[20] = np.random.beta(8, 2)  # High srv error rate
        
        # Protocol anomalies
        features[5] = np.random.choice([1, 17, 255])  # TCP, UDP, or unknown
        
        return features
    
    def _generate_mitm_features(self):
        """Generate features for MITM attack"""
        features = self._generate_normal_features(np.random.choice(list(self.iot_device_profiles.keys())))
        
        # Modify for MITM characteristics
        features[0] *= np.random.uniform(1.5, 3.0)  # Larger packets
        features[11] = np.random.beta(8, 2)  # High diff_srv_rate
        features[16] = np.random.beta(8, 2)  # High dst_host_diff_srv_rate
        features[23] = np.random.uniform(6, 8)  # High entropy (encrypted)
        
        # Protocol mixing
        if np.random.random() > 0.5:
            features[5] = np.random.choice([0, 1, 2, 7])  # Mixed protocols
        
        # Timing anomalies
        features[2] *= np.random.uniform(0.5, 2)  # Variable duration
        
        return features
    
    def _generate_eavesdropping_features(self):
        """Generate features for eavesdropping attack"""
        features = self._generate_normal_features(np.random.choice(list(self.iot_device_profiles.keys())))
        
        # Modify for eavesdropping characteristics
        features[2] *= 2  # Longer sessions
        features[10] = np.random.beta(9, 1)  # Very high same_srv_rate
        features[15] = np.random.beta(9, 1)  # Very high dst_host_same_srv_rate
        
        # Stealth indicators
        features[1] = np.random.poisson(3)  # Low packet count
        features[19] = np.random.beta(1, 9)  # Low error rate
        features[20] = np.random.beta(1, 9)  # Low srv error rate
        
        return features
    
    def _generate_data_injection_features(self):
        """Generate features for data injection attack"""
        features = self._generate_normal_features(np.random.choice(list(self.iot_device_profiles.keys())))
        
        # Modify for data injection characteristics
        features[0] *= np.random.uniform(2, 5)  # Much larger packets
        features[3] *= np.random.uniform(3, 8)  # High src_bytes
        features[4] *= np.random.uniform(3, 8)  # High dst_bytes
        
        # Payload anomalies
        features[23] = np.random.uniform(0, 4)  # Low entropy (structured data)
        
        # Protocol violations
        features[7] = np.random.choice([0, 1, 2])  # Unusual flags
        
        return features
    
    def _generate_replay_attack_features(self):
        """Generate features for replay attack"""
        features = self._generate_normal_features(np.random.choice(list(self.iot_device_profiles.keys())))
        
        # Modify for replay characteristics
        # Duplicate-like patterns
        features[1] = np.random.poisson(2)  # Low but consistent
        
        # Timing patterns (replays often have regular intervals)
        features[2] = np.random.normal(5, 0.5)  # Very consistent duration
        
        # Sequence anomalies
        features[10] = np.random.beta(5, 5)  # Mixed same_srv_rate
        features[11] = np.random.beta(5, 5)  # Mixed diff_srv_rate
        
        return features
    
    def simulate_dos_attack(self, n_packets=100, attack_subtype='UDP Flood'):
        """Simulate a specific DoS attack"""
        print(f"Simulating {attack_subtype} DoS attack with {n_packets} packets...")
        
        features = []
        attack_info = []
        
        for i in range(n_packets):
            feat = self._generate_dos_features()
            features.append(feat)
            attack_info.append({
                'type': 'dos',
                'subtype': attack_subtype,
                'packet_num': i + 1,
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'features': np.array(features),
            'attack_info': attack_info,
            'attack_type': 'dos',
            'subtype': attack_subtype,
            'total_packets': n_packets
        }
    
    def simulate_mitm_attack(self, n_packets=100, attack_subtype='ARP Spoofing'):
        """Simulate a specific MITM attack"""
        print(f"Simulating {attack_subtype} MITM attack with {n_packets} packets...")
        
        features = []
        attack_info = []
        
        for i in range(n_packets):
            feat = self._generate_mitm_features()
            features.append(feat)
            attack_info.append({
                'type': 'mitm',
                'subtype': attack_subtype,
                'packet_num': i + 1,
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'features': np.array(features),
            'attack_info': attack_info,
            'attack_type': 'mitm',
            'subtype': attack_subtype,
            'total_packets': n_packets
        }
    
    def simulate_eavesdropping_attack(self, n_packets=100, attack_subtype='Packet Sniffing'):
        """Simulate a specific eavesdropping attack"""
        print(f"Simulating {attack_subtype} attack with {n_packets} packets...")
        
        features = []
        attack_info = []
        
        for i in range(n_packets):
            feat = self._generate_eavesdropping_features()
            features.append(feat)
            attack_info.append({
                'type': 'eavesdropping',
                'subtype': attack_subtype,
                'packet_num': i + 1,
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'features': np.array(features),
            'attack_info': attack_info,
            'attack_type': 'eavesdropping',
            'subtype': attack_subtype,
            'total_packets': n_packets
        }
    
    def simulate_data_injection_attack(self, n_packets=100, attack_subtype='SQL Injection'):
        """Simulate a specific data injection attack"""
        print(f"Simulating {attack_subtype} attack with {n_packets} packets...")
        
        features = []
        attack_info = []
        
        for i in range(n_packets):
            feat = self._generate_data_injection_features()
            features.append(feat)
            attack_info.append({
                'type': 'data_injection',
                'subtype': attack_subtype,
                'packet_num': i + 1,
                'timestamp': datetime.now().isoformat()
            })
        
        return {
            'features': np.array(features),
            'attack_info': attack_info,
            'attack_type': 'data_injection',
            'subtype': attack_subtype,
            'total_packets': n_packets
        }
    
    def start_continuous_simulation(self, interval_seconds=5, batch_size=50):
        """
        Start continuous traffic simulation
        
        Args:
            interval_seconds: Time between batches
            batch_size: Number of samples per batch
        
        Yields:
            Batches of simulated traffic
        """
        print(f"Starting continuous simulation (batch size: {batch_size}, interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            batch_num = 1
            while True:
                print(f"\n[Batch {batch_num}] Generating traffic...")
                
                # Generate batch with random attack probability (0-40%)
                attack_prob = np.random.uniform(0, 0.4)
                batch = self.generate_live_traffic(batch_size, attack_prob)
                
                # Add batch number
                batch['batch_num'] = batch_num
                batch['attack_probability'] = attack_prob
                
                yield batch
                
                batch_num += 1
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\nSimulation stopped by user")
    
    def get_attack_statistics(self):
        """Get statistics about defined attacks"""
        stats = {
            'total_attack_types': len(self.attack_patterns),
            'attack_details': {}
        }
        
        for attack_type, details in self.attack_patterns.items():
            stats['attack_details'][attack_type] = {
                'name': details['name'],
                'subtypes_count': len(details['subtypes']),
                'indicators': details['indicators'],
                'intensity': details['intensity']
            }
        
        return stats