import os
import requests
import zipfile
import tarfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DatasetDownloader:
    def __init__(self, data_dir='data/real_datasets'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Define available datasets
        self.datasets = {
            'cicids2017': {
                'name': 'CICIDS2017',
                'description': 'Comprehensive intrusion detection dataset',
                'size': '16GB',
                'features': 80,
                'samples': '2.8M',
                'attack_types': ['DoS', 'DDoS', 'Brute Force', 'Infiltration', 'Botnet'],
                'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
                'files': ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv']
            },
            'nbaiot': {
                'name': 'N-BaIoT',
                'description': 'Network traffic of IoT devices infected by botnets',
                'size': '8GB',
                'features': 115,
                'samples': '7M',
                'attack_types': ['Mirai', 'Bashlite'],
                'url': 'https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT',
                'devices': ['Danmini', 'Ecobee', 'Ennio', 'Philips B120N10', 'Provision PT737E', 'Samsung SNH1011']
            },
            'iot23': {
                'name': 'IoT-23',
                'description': 'Labeled IoT network traffic with malware',
                'size': '20GB',
                'features': 'Variable',
                'samples': '3.6M',
                'attack_types': ['DDoS', 'DoS', 'Command & Control', 'Data exfiltration'],
                'url': 'https://www.stratosphereips.org/datasets-iot23',
                'scenarios': 20
            },
            'ton_iot': {
                'name': 'ToN-IoT',
                'description': 'Telemetry data of IoT services with attacks',
                'size': '5GB',
                'features': 44,
                'samples': '461K',
                'attack_types': ['Backdoor', 'DDoS', 'Injection', 'MITM', 'Password', 'Ransomware', 'Scanning', 'XSS'],
                'url': 'https://research.unsw.edu.au/projects/toniot-datasets'
            }
        }
    
    def list_available_datasets(self):
        """List all available datasets"""
        print("="*70)
        print("AVAILABLE IOT/IDS DATASETS")
        print("="*70)
        
        for dataset_id, info in self.datasets.items():
            print(f"\n📊 {info['name']} ({dataset_id}):")
            print(f"   • Description: {info['description']}")
            print(f"   • Size: {info['size']}")
            print(f"   • Features: {info['features']}")
            print(f"   • Samples: {info['samples']}")
            print(f"   • Attack types: {', '.join(info['attack_types'][:3])}...")
            print(f"   • URL: {info['url']}")
        
        print("\n" + "="*70)
        return self.datasets
    
    def download_simulated_fallback(self, dataset_id, n_samples=10000):
        """
        Generate simulated data as a fallback when real dataset is not available
        """
        print(f"Generating simulated data for {dataset_id} ({n_samples} samples)...")
        
        # Import data loader to generate simulated data
        try:
            from data.data_loader_enhanced import EnhancedDataLoader
            
            data_loader = EnhancedDataLoader()
            X, y, feature_names = data_loader.load_enhanced_dataset(
                n_samples=n_samples,
                attack_ratio=0.2
            )
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            df['label'] = y
            
            # Save to CSV
            output_path = f"{self.data_dir}/{dataset_id}_simulated.csv"
            df.to_csv(output_path, index=False)
            
            print(f"✅ Simulated dataset saved to: {output_path}")
            print(f"   • Samples: {len(df)}")
            print(f"   • Features: {len(feature_names)}")
            print(f"   • Attacks: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
            
            return df, feature_names
            
        except Exception as e:
            print(f"❌ Error generating simulated data: {e}")
            
            # Create simple dataset as last resort
            print("Creating simple dataset as fallback...")
            
            # Generate simple features
            np.random.seed(42)
            n_features = 24
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            
            # Add some pattern to attacks
            attack_indices = np.where(y == 1)[0]
            for idx in attack_indices:
                X[idx, 1] += 3  # High packet count
                X[idx, 7] += 2  # High error rate
            
            feature_names = [f'feature_{i}' for i in range(n_features)]
            
            df = pd.DataFrame(X, columns=feature_names)
            df['label'] = y
            
            output_path = f"{self.data_dir}/{dataset_id}_simple.csv"
            df.to_csv(output_path, index=False)
            
            print(f"✅ Simple dataset saved to: {output_path}")
            
            return df, feature_names
    
    def download_dataset(self, dataset_id, force_download=False):
        """
        Download or prepare a dataset
        
        Note: Most datasets require manual download due to size.
        This function provides instructions and generates simulated data.
        """
        if dataset_id not in self.datasets:
            print(f"❌ Dataset '{dataset_id}' not found. Available: {list(self.datasets.keys())}")
            return None
        
        dataset_info = self.datasets[dataset_id]
        
        print(f"\n📥 Preparing dataset: {dataset_info['name']}")
        print("="*50)
        
        # Check if already downloaded
        expected_files = [
            f"{self.data_dir}/{dataset_id}_processed.csv",
            f"{self.data_dir}/{dataset_id}_simulated.csv",
            f"{self.data_dir}/{dataset_id}_simple.csv"
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path) and not force_download:
                print(f"✅ Dataset already exists at: {file_path}")
                df = pd.read_csv(file_path)
                print(f"   • Loaded {len(df)} samples with {df.shape[1]-1} features")
                return df
        
        # Provide download instructions
        print(f"\n📋 DOWNLOAD INSTRUCTIONS for {dataset_info['name']}:")
        print("-" * 50)
        print(f"1. Visit: {dataset_info['url']}")
        print(f"2. Download the dataset (Size: {dataset_info['size']})")
        print(f"3. Extract to: {self.data_dir}/")
        print(f"4. Run preprocessing script")
        print("\n⏳ Since manual download is required, generating simulated data instead...")
        
        # Generate simulated data
        return self.download_simulated_fallback(dataset_id)
    
    def preprocess_cicids2017(self, filepath):
        """Preprocess CICIDS2017 dataset (if available)"""
        print("Preprocessing CICIDS2017 dataset...")
        
        try:
            # Load the dataset
            df = pd.read_csv(filepath)
            
            # Basic preprocessing
            print(f"   • Original shape: {df.shape}")
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Convert labels
            if 'Label' in df.columns:
                # Convert attack labels to binary
                df['label'] = df['Label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)
                df = df.drop('Label', axis=1)
            
            # Select subset of features (for manageability)
            if df.shape[1] > 50:
                # Select most important features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 30:
                    # Use correlation with label to select features
                    if 'label' in df.columns:
                        correlations = df[numeric_cols].corrwith(df['label']).abs()
                        top_features = correlations.nlargest(30).index
                        df = df[list(top_features) + ['label']]
            
            print(f"   • Processed shape: {df.shape}")
            print(f"   • Attack percentage: {df['label'].mean()*100:.1f}%")
            
            # Save processed version
            output_path = f"{self.data_dir}/cicids2017_processed.csv"
            df.to_csv(output_path, index=False)
            
            print(f"✅ Processed dataset saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing CICIDS2017: {e}")
            return None
    
    def create_training_test_split(self, df, test_size=0.2):
        """Create training and test splits"""
        from sklearn.model_selection import train_test_split
        
        if 'label' not in df.columns:
            print("❌ Dataset doesn't have 'label' column")
            return None
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Save splits
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_path = f"{self.data_dir}/train_split.csv"
        test_path = f"{self.data_dir}/test_split.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Training set ({len(train_df)} samples) saved to: {train_path}")
        print(f"✅ Test set ({len(test_df)} samples) saved to: {test_path}")
        
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'y_train': y_train.values,
            'y_test': y_test.values,
            'feature_names': list(X.columns)
        }
    
    def load_preprocessed_dataset(self, dataset_id='cicids2017'):
        """Load a preprocessed dataset"""
        processed_path = f"{self.data_dir}/{dataset_id}_processed.csv"
        
        if os.path.exists(processed_path):
            print(f"Loading preprocessed dataset from: {processed_path}")
            df = pd.read_csv(processed_path)
            return df
        else:
            print(f"❌ Processed dataset not found at: {processed_path}")
            print("Generating simulated data instead...")
            return self.download_simulated_fallback(dataset_id)[0]
    
    def generate_dataset_report(self, df, dataset_name):
        """Generate a report about the dataset"""
        report = {
            'dataset_name': dataset_name,
            'total_samples': len(df),
            'total_features': df.shape[1] - 1 if 'label' in df.columns else df.shape[1],
            'attack_samples': int(df['label'].sum()) if 'label' in df.columns else 0,
            'normal_samples': len(df) - (int(df['label'].sum()) if 'label' in df.columns else 0),
            'attack_percentage': (df['label'].mean() * 100) if 'label' in df.columns else 0,
            'missing_values': df.isnull().sum().sum(),
            'data_types': str(df.dtypes.value_counts().to_dict())
        }
        
        print(f"\n📊 DATASET REPORT: {dataset_name}")
        print("="*50)
        for key, value in report.items():
            print(f"   • {key.replace('_', ' ').title()}: {value}")
        
        # Save report
        report_path = f"{self.data_dir}/{dataset_name}_report.txt"
        with open(report_path, 'w') as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Report saved to: {report_path}")
        
        return report

# Example usage
if __name__ == "__main__":
    downloader = DatasetDownloader()
    
    # List available datasets
    downloader.list_available_datasets()
    
    # Download/generate a dataset
    df = downloader.download_dataset('cicids2017')
    
    if df is not None:
        # Generate report
        downloader.generate_dataset_report(df, 'cicids2017_simulated')
        
        # Create train/test split
        splits = downloader.create_training_test_split(df)