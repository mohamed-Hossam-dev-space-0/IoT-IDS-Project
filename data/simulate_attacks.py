import numpy as np
import pandas as pd
from scapy.all import *
import random
import time

class AttackSimulator:
    def __init__(self):
        self.normal_patterns = self._define_normal_patterns()
        self.attack_patterns = self._define_attack_patterns()
    
    def _define_normal_patterns(self):
        """Define normal IoT traffic patterns"""
        return {
            'packet_size': {'mean': 500, 'std': 150},
            'packet_interval': {'mean': 1.0, 'std': 0.3},
            'protocol_dist': {'TCP': 0.6, 'UDP': 0.3, 'ICMP': 0.1}
        }
    
    def _define_attack_patterns(self):
        """Define attack traffic patterns"""
        return {
            'dos': {
                'packet_size': {'mean': 100, 'std': 50},
                'packet_interval': {'mean': 0.01, 'std': 0.005},
                'protocol_dist': {'TCP': 0.8, 'UDP': 0.15, 'ICMP': 0.05}
            },
            'mitm': {
                'packet_size': {'mean': 800, 'std': 200},
                'packet_interval': {'mean': 0.5, 'std': 0.2},
                'protocol_dist': {'TCP': 0.4, 'UDP': 0.4, 'ICMP': 0.2}
            }
        }
    
    def analyze_dos_attack(self):
        """Analyze DoS attack characteristics"""
        return {
            "Attack Type": "Denial of Service (DoS)",
            "Mechanism": "Flood target with excessive requests",
            "Target": "Network bandwidth & device resources",
            "Indicators": [
                "Abnormally high packet rate",
                "Small packet sizes",
                "Spoofed source IPs",
                "Concentrated destination IP"
            ],
            "Impact": "Service disruption, resource exhaustion",
            "Detection Method": "Rate limiting, anomaly detection"
        }
    
    def analyze_mitm_attack(self):
        """Analyze MITM attack characteristics"""
        return {
            "Attack Type": "Man-in-the-Middle (MITM)",
            "Mechanism": "Intercept and modify communications",
            "Target": "Data integrity & confidentiality",
            "Indicators": [
                "Unusual protocol combinations",
                "Duplicate packets",
                "SSL/TLS certificate issues",
                "Unexpected routing changes"
            ],
            "Impact": "Data theft, session hijacking",
            "Detection Method": "Certificate validation, timing analysis"
        }
    
    def simulate_dos_attack(self, n_packets=100):
        """
        Simulate DoS attack traffic
        """
        print(f"Simulating DoS attack with {n_packets} packets...")
        
        features = []
        pattern = self.attack_patterns['dos']
        
        for _ in range(n_packets):
            # Generate DoS-like features
            packet_features = [
                np.random.normal(pattern['packet_size']['mean'], 
                               pattern['packet_size']['std']),  # packet_size
                self._get_protocol_code(pattern['protocol_dist']),  # protocol
                np.random.exponential(0.1),  # duration (very short)
                np.random.lognormal(4, 1),   # src_bytes
                np.random.lognormal(3, 1),   # dst_bytes
                n_packets // 10,             # high count
                n_packets // 20,             # high srv_count
                np.random.beta(8, 2),        # high serror_rate
                np.random.beta(8, 2),        # high srv_serror_rate
                np.random.beta(1, 9),        # rerror_rate
                np.random.beta(1, 9),        # srv_rerror_rate
                np.random.beta(1, 9),        # same_srv_rate (low)
                np.random.beta(8, 2),        # diff_srv_rate (high)
                np.random.beta(8, 2),        # srv_diff_host_rate (high)
                np.random.poisson(100),      # very high dst_host_count
                np.random.poisson(50),       # very high dst_host_srv_count
                np.random.beta(1, 9),        # dst_host_same_srv_rate (low)
                np.random.beta(8, 2),        # dst_host_diff_srv_rate (high)
                np.random.beta(1, 9),        # dst_host_same_src_port_rate
                np.random.beta(8, 2),        # dst_host_srv_diff_host_rate
                np.random.beta(8, 2),        # high dst_host_serror_rate
                np.random.beta(8, 2),        # high dst_host_srv_serror_rate
                np.random.beta(1, 9),        # dst_host_rerror_rate
                np.random.beta(1, 9),         # dst_host_srv_rerror_rate
            ]
            features.append(packet_features)
        
        return np.array(features)
    
    def simulate_mitm_attack(self, n_packets=100):
        """
        Simulate MITM attack traffic
        """
        print(f"Simulating MITM attack with {n_packets} packets...")
        
        features = []
        pattern = self.attack_patterns['mitm']
        
        for _ in range(n_packets):
            # Generate MITM-like features
            packet_features = [
                np.random.normal(pattern['packet_size']['mean'], 
                               pattern['packet_size']['std']),  # larger packets
                self._get_protocol_code(pattern['protocol_dist']),  # mixed protocols
                np.random.exponential(2.0),   # longer duration
                np.random.lognormal(7, 1),    # high src_bytes
                np.random.lognormal(7, 1),    # high dst_bytes
                np.random.poisson(2),         # normal count
                np.random.poisson(1),         # normal srv_count
                np.random.beta(5, 5),         # medium serror_rate
                np.random.beta(5, 5),         # medium srv_serror_rate
                np.random.beta(5, 5),         # medium rerror_rate
                np.random.beta(5, 5),         # medium srv_rerror_rate
                np.random.beta(5, 5),         # mixed same_srv_rate
                np.random.beta(5, 5),         # mixed diff_srv_rate
                np.random.beta(5, 5),         # mixed srv_diff_host_rate
                np.random.poisson(5),         # normal dst_host_count
                np.random.poisson(3),         # normal dst_host_srv_count
                np.random.beta(5, 5),         # mixed dst_host_same_srv_rate
                np.random.beta(5, 5),         # mixed dst_host_diff_srv_rate
                np.random.beta(8, 2),         # high dst_host_same_src_port_rate
                np.random.beta(8, 2),         # high dst_host_srv_diff_host_rate
                np.random.beta(5, 5),         # medium dst_host_serror_rate
                np.random.beta(5, 5),         # medium dst_host_srv_serror_rate
                np.random.beta(5, 5),         # medium dst_host_rerror_rate
                np.random.beta(5, 5),         # medium dst_host_srv_rerror_rate
            ]
            features.append(packet_features)
        
        return np.array(features)
    
    def simulate_normal_traffic(self, n_packets=100):
        """
        Simulate normal IoT traffic
        """
        print(f"Simulating normal traffic with {n_packets} packets...")
        
        features = []
        pattern = self.normal_patterns
        
        for _ in range(n_packets):
            packet_features = [
                np.random.normal(pattern['packet_size']['mean'], 
                               pattern['packet_size']['std']),
                self._get_protocol_code(pattern['protocol_dist']),
                np.random.exponential(pattern['packet_interval']['mean']),
                np.random.lognormal(6, 1),
                np.random.lognormal(6, 1),
                np.random.poisson(5),
                np.random.poisson(3),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
                np.random.beta(8, 2),
                np.random.beta(2, 8),
                np.random.beta(2, 8),
                np.random.poisson(10),
                np.random.poisson(6),
                np.random.beta(8, 2),
                np.random.beta(2, 8),
                np.random.beta(7, 3),
                np.random.beta(3, 7),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
                np.random.beta(1, 9),
            ]
            features.append(packet_features)
        
        return np.array(features)
    
    def simulate_multiple_attacks(self, n_samples=200):
        """
        Simulate mix of normal and attack traffic
        """
        normal = self.simulate_normal_traffic(n_samples // 2)
        dos = self.simulate_dos_attack(n_samples // 4)
        mitm = self.simulate_mitm_attack(n_samples // 4)
        
        combined = np.vstack([normal, dos, mitm])
        
        # Create labels (0: normal, 1: attack)
        labels = np.zeros(len(combined))
        labels[len(normal):] = 1  # Last half are attacks
        
        print(f"Generated {len(normal)} normal, {len(dos)} DoS, {len(mitm)} MITM samples")
        
        return combined
    
    def _get_protocol_code(self, protocol_dist):
        """Convert protocol distribution to code"""
        protocols = list(protocol_dist.keys())
        probs = list(protocol_dist.values())
        protocol = np.random.choice(protocols, p=probs)
        
        # Map to codes: TCP=0, UDP=1, ICMP=2
        mapping = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
        return mapping.get(protocol, 0)