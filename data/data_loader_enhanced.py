import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataLoader:
    def __init__(self):
        self.scaler = None
        self.feature_names = self._get_feature_names()
        print("EnhancedDataLoader initialized")
    
    def _get_feature_names(self):
        """Return comprehensive feature names"""
        return [
            'packet_size', 'packet_count', 'flow_duration', 'src_bytes', 'dst_bytes',
            'protocol', 'service', 'flag', 'src_count', 'dst_count', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'packet_interval_mean'
        ]
    
    def load_enhanced_dataset(self, n_samples=5000, attack_ratio=0.2):
        """
        Generate enhanced IoT dataset with realistic patterns
        """
        print(f"Generating enhanced IoT dataset with {n_samples} samples...")
        
        np.random.seed(42)
        
        # Initialize data dictionary
        data = {}
        
        # 1. Basic network features
        data['packet_size'] = self._generate_packet_sizes(n_samples)
        data['packet_count'] = self._generate_packet_counts(n_samples)
        data['flow_duration'] = self._generate_flow_durations(n_samples)
        data['src_bytes'] = np.random.lognormal(8, 1.5, n_samples)
        data['dst_bytes'] = np.random.lognormal(8, 1.5, n_samples)
        
        # 2. Protocol and service features
        data['protocol'] = np.random.choice([0, 1, 2, 6, 17], n_samples, 
                                           p=[0.4, 0.1, 0.05, 0.4, 0.05])
        
        # 3. Statistical features
        data['src_count'] = np.random.poisson(5, n_samples)
        data['dst_count'] = np.random.poisson(5, n_samples)
        data['same_srv_rate'] = np.random.beta(8, 2, n_samples)
        data['diff_srv_rate'] = np.random.beta(2, 8, n_samples)
        data['srv_diff_host_rate'] = np.random.beta(3, 7, n_samples)
        
        # 4. Host-based features
        data['dst_host_count'] = np.random.poisson(10, n_samples)
        data['dst_host_srv_count'] = np.random.poisson(6, n_samples)
        data['dst_host_same_srv_rate'] = np.random.beta(7, 3, n_samples)
        data['dst_host_diff_srv_rate'] = np.random.beta(3, 7, n_samples)
        data['dst_host_same_src_port_rate'] = np.random.beta(6, 4, n_samples)
        data['dst_host_srv_diff_host_rate'] = np.random.beta(4, 6, n_samples)
        
        # 5. Error rates
        data['dst_host_serror_rate'] = np.random.beta(1, 9, n_samples)
        data['dst_host_srv_serror_rate'] = np.random.beta(1, 9, n_samples)
        data['dst_host_rerror_rate'] = np.random.beta(1, 9, n_samples)
        data['dst_host_srv_rerror_rate'] = np.random.beta(1, 9, n_samples)
        
        # 6. Timing features
        data['packet_interval_mean'] = np.random.exponential(0.5, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create labels
        labels = np.zeros(n_samples)
        n_attacks = int(n_samples * attack_ratio)
        attack_indices = np.random.choice(n_samples, n_attacks, replace=False)
        labels[attack_indices] = 1
        
        # Inject attack patterns
        self._inject_attack_patterns(df, attack_indices)
        
        # Shuffle
        X = df.values
        X, labels = shuffle(X, labels, random_state=42)
        
        print(f"✅ Dataset generated:")
        print(f"   • Samples: {X.shape[0]}")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Normal: {sum(labels == 0)}")
        print(f"   • Attacks: {sum(labels == 1)}")
        
        return X, labels, self.feature_names
    
    def _generate_packet_sizes(self, n_samples):
        """Generate realistic packet size distribution"""
        sizes = np.zeros(n_samples)
        
        # Mix of distributions
        normal_idx = int(n_samples * 0.7)
        sizes[:normal_idx] = np.random.normal(500, 150, normal_idx)
        
        large_idx = int(n_samples * 0.2)
        sizes[normal_idx:normal_idx+large_idx] = np.random.lognormal(7, 0.5, large_idx)
        
        small_idx = n_samples - normal_idx - large_idx
        sizes[normal_idx+large_idx:] = np.random.exponential(100, small_idx)
        
        return np.abs(sizes)
    
    def _generate_packet_counts(self, n_samples):
        """Generate realistic packet count distribution"""
        counts = np.zeros(n_samples)
        
        low_idx = int(n_samples * 0.8)
        counts[:low_idx] = np.random.poisson(5, low_idx)
        
        medium_idx = int(n_samples * 0.15)
        counts[low_idx:low_idx+medium_idx] = np.random.poisson(20, medium_idx)
        
        high_idx = n_samples - low_idx - medium_idx
        counts[low_idx+medium_idx:] = np.random.poisson(100, high_idx)
        
        return counts
    
    def _generate_flow_durations(self, n_samples):
        """Generate realistic flow durations"""
        durations = np.zeros(n_samples)
        
        short_idx = int(n_samples * 0.6)
        durations[:short_idx] = np.random.exponential(1, short_idx)
        
        medium_idx = int(n_samples * 0.3)
        durations[short_idx:short_idx+medium_idx] = np.random.exponential(10, medium_idx)
        
        long_idx = n_samples - short_idx - medium_idx
        durations[short_idx+medium_idx:] = np.random.exponential(60, long_idx)
        
        return durations
    
    def _inject_attack_patterns(self, df, attack_indices):
        """Inject realistic attack patterns"""
        n_attacks = len(attack_indices)
        
        # Split into attack types
        dos_idx = attack_indices[:int(n_attacks * 0.4)]
        mitm_idx = attack_indices[int(n_attacks * 0.4):int(n_attacks * 0.7)]
        remaining = attack_indices[int(n_attacks * 0.7):]
        
        # DoS Attack Patterns
        if len(dos_idx) > 0:
            df.loc[dos_idx, 'packet_count'] *= np.random.randint(50, 500, len(dos_idx))
            df.loc[dos_idx, 'packet_size'] = np.random.randint(64, 128, len(dos_idx))
            df.loc[dos_idx, 'src_count'] *= 10
            df.loc[dos_idx, 'dst_host_count'] *= 20
            df.loc[dos_idx, 'dst_host_serror_rate'] = np.random.beta(8, 2, len(dos_idx))
        
        # MITM Attack Patterns
        if len(mitm_idx) > 0:
            df.loc[mitm_idx, 'packet_size'] *= np.random.uniform(1.5, 3.0, len(mitm_idx))
            df.loc[mitm_idx, 'packet_interval_mean'] *= 3
        
        # Other attacks
        if len(remaining) > 0:
            df.loc[remaining, 'dst_bytes'] *= np.random.uniform(3, 8, len(remaining))
            df.loc[remaining, 'same_srv_rate'] = np.random.beta(1, 9, len(remaining))
    
    def load_simple_dataset(self, n_samples=1000):
        """Load a simple dataset for quick testing"""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42,
            weights=[0.8, 0.2]
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        return X, y, feature_names

# Test the data loader
if __name__ == "__main__":
    loader = EnhancedDataLoader()
    X, y, features = loader.load_enhanced_dataset(1000)
    print(f"\nTest successful! Generated {X.shape[0]} samples.")