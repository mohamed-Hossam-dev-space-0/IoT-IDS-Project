from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import joblib
import os
from .ids_model import IDSModel
from data.data_loader import DataLoader

class ModelTrainer:
    def __init__(self, model_type='rf', dataset='simulated'):
        """
        Initialize model trainer
        
        Args:
            model_type: Type of model to train
            dataset: Dataset to use
        """
        self.model_type = model_type
        self.dataset = dataset
        self.data_loader = DataLoader()
        self.model = None
        self.best_params = None
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess data based on dataset choice
        """
        if self.dataset == 'cicids2017':
            X, y = self.data_loader.load_cicids2017()
        elif self.dataset == 'nbaiot':
            X, y = self.data_loader.load_nbaiot()
        else:
            X, y = self.data_loader.load_simulated_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(X, y)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, hyperparameter_tuning=False):
        """
        Train the IDS model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Trained model and training history
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        
        # Initialize model
        input_shape = (X_train.shape[1], 1) if self.model_type in ['cnn', 'lstm'] else None
        self.model = IDSModel(model_type=self.model_type, input_shape=input_shape)
        
        if hyperparameter_tuning:
            self._perform_hyperparameter_tuning(X_train, y_train)
        
        # Train model
        self.model.train(X_train, y_train, X_val, y_val)
        
        return self.model, self.model.history
    
    def _perform_hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for traditional ML models
        """
        if self.model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
            
            # Update model with best parameters
            self.model.model = grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of evaluation results
        """
        print("\nEvaluating model performance...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, model.predict_proba(X_test)) if hasattr(model, 'predict_proba') else 'N/A'
        }
        
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'True_Negatives': tn,
            'False_Positives': fp,
            'False_Negatives': fn,
            'True_Positives': tp,
            'False_Positive_Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'False_Negative_Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'Classification_Report': report
        })
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        for key, value in metrics.items():
            if key != 'Classification_Report':
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
        
        print("\nClassification Report:")
        print(report)
        
        print(f"\nConfusion Matrix:")
        print(f"[[{tn} {fp}]")
        print(f" [{fn} {tp}]]")
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        if self.model_type in ['cnn', 'lstm']:
            print("Cross-validation not supported for neural networks in this implementation")
            return None
        
        # Initialize model
        model = IDSModel(model_type=self.model_type)
        
        # Perform cross-validation
        scores = cross_val_score(
            model.model, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, model, filename=None):
        """
        Save trained model to file
        
        Args:
            model: Trained model
            filename: Output filename
        
        Returns:
            Path to saved model
        """
        if filename is None:
            filename = f"ids_model_{self.model_type}.pkl"
        
        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        
        filepath = os.path.join(save_dir, filename)
        model.save(filepath)
        
        return filepath
    
    def load_model(self, filename=None):
        """
        Load trained model from file
        
        Args:
            filename: Model filename
        
        Returns:
            Loaded model
        """
        if filename is None:
            filename = f"ids_model_{self.model_type}.pkl"
        
        filepath = os.path.join("saved_models", filename)
        
        if os.path.exists(filepath):
            input_shape = (23, 1) if self.model_type in ['cnn', 'lstm'] else None
            model = IDSModel(model_type=self.model_type, input_shape=input_shape)
            model.load(filepath)
            return model
        else:
            print(f"Model file {filepath} not found")
            return None