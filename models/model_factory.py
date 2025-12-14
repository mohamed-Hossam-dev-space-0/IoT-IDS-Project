import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelFactory:
    def __init__(self):
        self.models = {}
        print("ModelFactory initialized - Create AI models for intrusion detection")
    
    def create_model(self, model_type='random_forest', **kwargs):
        """Create and return a model of specified type"""
        print(f"Creating {model_type} model...")
        
        if model_type == 'random_forest':
            return RandomForestModel(**kwargs)
        elif model_type == 'xgboost':
            return XGBoostModel(**kwargs)
        elif model_type == 'svm':
            return SVMModel(**kwargs)
        elif model_type == 'mlp':
            return MLPModel(**kwargs)
        else:
            print(f"Unknown model type: {model_type}. Using Random Forest.")
            return RandomForestModel(**kwargs)
    
    def create_ensemble(self, models_dict):
        """Create ensemble from multiple models"""
        return EnsembleModel(models_dict)

class BaseModel:
    def __init__(self, name="Base Model"):
        self.model = None
        self.name = name
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"  ✅ {self.name} trained on {X_train.shape[0]} samples")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[:, 1] = predictions
            proba[:, 0] = 1 - predictions
            return proba
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import time
        
        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'inference_time': inference_time / len(X_test) if len(X_test) > 0 else 0
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"  💾 Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"  📂 Model loaded from {filepath}")

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=10, **kwargs):
        super().__init__(name="Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            **kwargs
        )

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=6, **kwargs):
        super().__init__(name="XGBoost")
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                **kwargs
            )
        except ImportError:
            print("  ⚠️  XGBoost not installed. Using Random Forest instead.")
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            self.name = "Random Forest (XGBoost fallback)"

class SVMModel(BaseModel):
    def __init__(self, C=1.0, kernel='rbf', **kwargs):
        super().__init__(name="Support Vector Machine")
        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=42,
            **kwargs
        )

class MLPModel(BaseModel):
    def __init__(self, hidden_layers=(100, 50), **kwargs):
        super().__init__(name="Multi-Layer Perceptron")
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            **kwargs
        )

class EnsembleModel(BaseModel):
    def __init__(self, models_dict):
        super().__init__(name="Voting Ensemble")
        self.models_dict = models_dict
        self.model = None  # Ensemble doesn't have a single sklearn model
    
    def predict(self, X):
        """Make predictions using voting ensemble"""
        predictions = []
        
        for model_name, model_data in self.models_dict.items():
            if hasattr(model_data, 'predict'):
                pred = model_data.predict(X)
            elif isinstance(model_data, dict) and 'model' in model_data:
                pred = model_data['model'].predict(X)
            else:
                continue
            
            predictions.append(pred)
        
        if not predictions:
            raise ValueError("No valid models in ensemble")
        
        # Majority voting
        predictions_array = np.array(predictions)
        ensemble_pred = np.round(np.mean(predictions_array, axis=0)).astype(int)
        
        return ensemble_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import time
        
        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'inference_time': inference_time / len(X_test) if len(X_test) > 0 else 0
        }
        
        return metrics

# Test the model factory
if __name__ == "__main__":
    print("Testing ModelFactory...")
    
    # Create sample data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    factory = ModelFactory()
    
    # Test different models
    models_to_test = ['random_forest', 'svm', 'mlp']
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type}...")
        model = factory.create_model(model_type)
        model.train(X, y)
        
        # Make prediction
        test_point = np.array([[2, 3]])
        prediction = model.predict(test_point)
        print(f"  Prediction for {test_point[0]}: {prediction[0]}")
        
        # Evaluate
        metrics = model.evaluate(X, y)
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    print("\n✅ ModelFactory test complete!")