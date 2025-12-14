import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ids_model import IDSModel
from data.data_loader import DataLoader

class TestIDSModels(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.data_loader = DataLoader()
        X, y = self.data_loader.load_simulated_data(n_samples=100)
        self.X_train = X[:80]
        self.y_train = y[:80]
        self.X_test = X[80:]
        self.y_test = y[80:]
    
    def test_random_forest(self):
        """Test Random Forest model"""
        model = IDSModel(model_type='rf')
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_cnn_model(self):
        """Test CNN model"""
        input_shape = (self.X_train.shape[1], 1)
        model = IDSModel(model_type='cnn', input_shape=input_shape)
        model.train(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        model = IDSModel(model_type='rf')
        model.train(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        model = IDSModel(model_type='rf')
        model.train(self.X_train, self.y_train)
        importances = model.get_feature_importance()
        if importances is not None:
            self.assertEqual(len(importances), self.X_train.shape[1])
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Create and train model
        model = IDSModel(model_type='rf')
        model.train(self.X_train, self.y_train)
        
        # Save model
        model.save('test_model.pkl')
        
        # Load model
        loaded_model = IDSModel(model_type='rf')
        loaded_model.load('test_model.pkl')
        
        # Compare predictions
        original_pred = model.predict(self.X_test)
        loaded_pred = loaded_model.predict(self.X_test)
        
        self.assertTrue(np.array_equal(original_pred, loaded_pred))
        
        # Clean up
        import os
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')

class TestDataLoader(unittest.TestCase):
    def test_data_generation(self):
        """Test data generation"""
        data_loader = DataLoader()
        X, y = data_loader.load_simulated_data(n_samples=100)
        
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(y.shape[0], 100)
        self.assertEqual(X.shape[1], 24)  # 24 features
        self.assertIn(0, y)  # Should have normal samples
        self.assertIn(1, y)  # Should have attack samples
    
    def test_data_split(self):
        """Test data splitting"""
        data_loader = DataLoader()
        X, y = data_loader.load_simulated_data(n_samples=100)
        splits = data_loader.split_data(X, y)
        
        self.assertEqual(len(splits), 6)  # X_train, X_val, X_test, y_train, y_val, y_test
        
        # Check sizes
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        total = len(X)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), total)

if __name__ == '__main__':
    unittest.main()