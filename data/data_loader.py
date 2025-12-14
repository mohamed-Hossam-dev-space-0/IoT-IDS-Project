import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.feature_names = [
            'packet_size', 'protocol_type', 'duration', 'src_bytes', 'dst_bytes',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label'
        ]
    
    def get_feature_names(self):
        """Return feature names"""
        return self.feature_names[:-1]  # Exclude label
    
    def generate_simulated_data(self, n_samples=10000):
        """
        Generate simulated IoT network traffic data
        """
        np.random.seed(42)
        
        # Generate normal traffic features
        data = {
            'packet_size': np.random.normal(500, 150, n_samples),
            'protocol_type': np.random.choice([0, 1, 2], n_samples),  # 0:TCP, 1:UDP, 2:ICMP
            'duration': np.random.exponential(1.0, n_samples),
            'src_bytes': np.random.lognormal(6, 1, n_samples),
            'dst_bytes': np.random.lognormal(6, 1, n_samples),
            'count': np.random.poisson(5, n_samples),
            'srv_count': np.random.poisson(3, n_samples),
            'serror_rate': np.random.beta(1, 9, n_samples),
            'srv_serror_rate': np.random.beta(1, 9, n_samples),
            'rerror_rate': np.random.beta(1, 9, n_samples),
            'srv_rerror_rate': np.random.beta(1, 9, n_samples),
            'same_srv_rate': np.random.beta(8, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 8, n_samples),
            'srv_diff_host_rate': np.random.beta(2, 8, n_samples),
            'dst_host_count': np.random.poisson(10, n_samples),
            'dst_host_srv_count': np.random.poisson(6, n_samples),
            'dst_host_same_srv_rate': np.random.beta(8, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 8, n_samples),
            'dst_host_same_src_port_rate': np.random.beta(7, 3, n_samples),
            'dst_host_srv_diff_host_rate': np.random.beta(3, 7, n_samples),
            'dst_host_serror_rate': np.random.beta(1, 9, n_samples),
            'dst_host_srv_serror_rate': np.random.beta(1, 9, n_samples),
            'dst_host_rerror_rate': np.random.beta(1, 9, n_samples),
            'dst_host_srv_rerror_rate': np.random.beta(1, 9, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate labels (80% normal, 20% attack)
        n_attacks = int(0.2 * n_samples)
        labels = np.zeros(n_samples)
        labels[-n_attacks:] = 1  # Last 20% are attacks
        
        # Modify attack samples to have different characteristics
        attack_indices = np.where(labels == 1)[0]
        
        # DoS attacks: high packet count, high error rates
        df.loc[attack_indices[:len(attack_indices)//2], 'count'] *= 10
        df.loc[attack_indices[:len(attack_indices)//2], 'serror_rate'] = np.random.beta(8, 2, len(attack_indices)//2)
        
        # MITM attacks: unusual protocol combinations, different patterns
        df.loc[attack_indices[len(attack_indices)//2:], 'protocol_type'] = np.random.choice([0, 1, 2], len(attack_indices)//2, p=[0.1, 0.1, 0.8])
        df.loc[attack_indices[len(attack_indices)//2:], 'diff_srv_rate'] = np.random.beta(8, 2, len(attack_indices)//2)
        
        df['label'] = labels
        
        return shuffle(df, random_state=42)
    
    def load_simulated_data(self):
        """
        Load and return simulated dataset
        """
        print("Generating simulated IoT network data...")
        df = self.generate_simulated_data(10000)
        
        # Separate features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Normal samples: {sum(y == 0)}")
        print(f"Attack samples: {sum(y == 1)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_relative_size, 
            random_state=42, 
            stratify=y_train_val
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_cicids2017(self):
        """
        Load CICIDS2017 dataset (placeholder - requires actual dataset)
        """
        print("Note: CICIDS2017 dataset not included. Using simulated data instead.")
        return self.load_simulated_data()
    
    def load_nbaiot(self):
        """
        Load N-BaIoT dataset (placeholder - requires actual dataset)
        """
        print("Note: N-BaIoT dataset not included. Using simulated data instead.")
        return self.load_simulated_data()