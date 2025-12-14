import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class IDSEvaluator:
    def __init__(self, model, X_test, y_test):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_proba = None
        
    def calculate_all_metrics(self):
        """
        Calculate all evaluation metrics
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            self.y_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'F1_Score': f1_score(self.y_test, self.y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities are available
        if self.y_proba is not None:
            metrics['ROC_AUC'] = roc_auc_score(self.y_test, self.y_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'True_Negatives': tn,
            'False_Positives': fp,
            'False_Negatives': fn,
            'True_Positives': tp,
            'False_Positive_Rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'False_Negative_Rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'Detection_Rate': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
        
        return metrics
    
    def generate_classification_report(self):
        """
        Generate detailed classification report
        
        Returns:
            Classification report as string
        """
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=['Normal', 'Attack'],
            digits=4
        )
        
        return report
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            save_path: Path to save the plot
        """
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve
        
        Args:
            save_path: Path to save the plot
        """
        if not hasattr(self.model, 'predict_proba'):
            print("ROC curve not available - model doesn't support probability predictions")
            return
        
        from sklearn.metrics import roc_curve
        
        self.y_proba = self.model.predict_proba(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            save_path: Path to save the plot
        """
        if not hasattr(self.model, 'predict_proba'):
            print("Precision-Recall curve not available")
            return
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        self.y_proba = self.model.predict_proba(self.X_test)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        avg_precision = average_precision_score(self.y_test, self.y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def get_detection_statistics(self):
        """
        Get detailed detection statistics
        
        Returns:
            Dictionary of detection statistics
        """
        if self.y_pred is None:
            self.y_pred = self.model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        stats = {
            'total_samples': len(self.y_test),
            'actual_attacks': sum(self.y_test == 1),
            'actual_normal': sum(self.y_test == 0),
            'detected_attacks': tp,
            'missed_attacks': fn,
            'false_alarms': fp,
            'correctly_classified_normal': tn,
            'attack_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'overall_accuracy': (tp + tn) / len(self.y_test)
        }
        
        return stats