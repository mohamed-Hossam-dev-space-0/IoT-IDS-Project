import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import joblib
import os

class IDSModel:
    def __init__(self, model_type='rf', input_shape=None):
        """
        Initialize IDS model
        
        Args:
            model_type: 'rf' (Random Forest), 'svm', 'mlp', 'cnn', 'lstm', 'ensemble'
            input_shape: Input shape for neural networks
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=100,
                random_state=42
            )
        
        elif self.model_type == 'cnn':
            if self.input_shape is None:
                raise ValueError("Input shape must be specified for CNN")
            
            self.model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=self.input_shape),
                Dropout(0.2),
                Conv1D(32, 3, activation='relu'),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )
        
        elif self.model_type == 'lstm':
            if self.input_shape is None:
                raise ValueError("Input shape must be specified for LSTM")
            
            self.model = Sequential([
                LSTM(64, return_sequences=True, input_shape=self.input_shape),
                Dropout(0.2),
                LSTM(32),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )
        
        elif self.model_type == 'ensemble':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"Training {self.model_type.upper()} model...")
        
        if self.model_type in ['cnn', 'lstm']:
            # Reshape for neural networks
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                validation_data = (X_val, y_val)
            else:
                validation_data = None
            
            # Train neural network
            self.history = self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_data=validation_data,
                verbose=1
            )
        
        else:
            # Train traditional ML model
            self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions (0 for normal, 1 for attack)
        """
        if self.model_type in ['cnn', 'lstm']:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            predictions = self.model.predict(X)
            return (predictions > 0.5).astype(int).flatten()
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
        
        Returns:
            Probability scores
        """
        if self.model_type in ['cnn', 'lstm']:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return self.model.predict(X).flatten()
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type in ['cnn', 'lstm']:
            self.model.save(filepath)
        else:
            joblib.dump(self.model, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        if self.model_type in ['cnn', 'lstm']:
            from tensorflow.keras.models import load_model
            self.model = load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        
        print(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models)
        
        Returns:
            Feature importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("Feature importance not available for this model type")
            return None