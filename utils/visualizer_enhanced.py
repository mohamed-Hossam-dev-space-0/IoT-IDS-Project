import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedVisualizer:
    def __init__(self):
        self.output_dir = "outputs/graphs"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"EnhancedVisualizer initialized - Outputs to: {self.output_dir}")
    
    def plot_model_comparison(self, model_results):
        """Create professional model comparison visualization"""
        print("Generating model comparison visualization...")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main comparison bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [model_results[m].get(metric, 0) for m in models]
            ax1.bar(x + i*width - width*1.5, values, width, 
                   label=metric.replace('_', ' ').title(),
                   color=color, alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m[:15] for m in models], rotation=45, ha='right')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.05)
        
        # Add value labels
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                value = model_results[model].get(metric, 0)
                if value > 0:
                    ax1.text(i + j*width - width*1.5, value + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', 
                            fontsize=8, rotation=90)
        
        # 2. F1-Score comparison
        ax2 = fig.add_subplot(gs[0, 2])
        f1_scores = [model_results[m].get('f1_score', 0) for m in models]
        
        bars = ax2.barh(models, f1_scores, color='steelblue', alpha=0.7)
        ax2.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
        ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()  # Highest score at top
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 1.05)
        
        # Add score values
        for bar, score in zip(bars, f1_scores):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        # 3. Training time comparison (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        train_times = []
        valid_models = []
        
        for model in models:
            if 'train_time' in model_results[model]:
                train_times.append(model_results[model]['train_time'])
                valid_models.append(model)
        
        if train_times:
            bars = ax3.bar(valid_models, train_times, color='lightcoral', alpha=0.7)
            ax3.set_xlabel('Models', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
            ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
            ax3.set_xticklabels([m[:10] for m in valid_models], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add time labels
            for bar, time_val in zip(bars, train_times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Training times not available',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color='gray')
            ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        
        # 4. Precision-Recall scatter
        ax4 = fig.add_subplot(gs[1, 1])
        precisions = [model_results[m].get('precision', 0) for m in models]
        recalls = [model_results[m].get('recall', 0) for m in models]
        
        scatter = ax4.scatter(precisions, recalls, s=200, 
                             c=range(len(models)), cmap='viridis',
                             alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add model labels
        for i, model in enumerate(models):
            ax4.annotate(model[:12], (precisions[i], recalls[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        ax4.set_xlabel('Precision', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Recall', fontsize=10, fontweight='bold')
        ax4.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1.05)
        ax4.set_ylim(0, 1.05)
        
        # Add diagonal reference line
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        
        # 5. Model ranking by composite score
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Calculate composite score
        composite_scores = {}
        for model_name, results in model_results.items():
            score = (results.get('accuracy', 0) * 0.25 +
                    results.get('precision', 0) * 0.25 +
                    results.get('recall', 0) * 0.25 +
                    results.get('f1_score', 0) * 0.25)
            composite_scores[model_name] = score
        
        # Sort by score
        sorted_models = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        model_names = [m[0][:15] for m in sorted_models]
        scores = [m[1] for m in sorted_models]
        
        bars = ax5.barh(model_names, scores, color='lightseagreen', alpha=0.7)
        ax5.set_xlabel('Composite Score', fontsize=10, fontweight='bold')
        ax5.set_title('Model Ranking', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()  # Highest score at top
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.set_xlim(0, 1.05)
        
        # Add score values
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        plt.suptitle('AI Model Performance Analysis for IoT Intrusion Detection', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Model comparison saved to: {save_path}")
        return save_path
    
    def plot_confusion_matrix_enhanced(self, y_true, y_pred, model_name="Model"):
        """Plot enhanced confusion matrix with metrics"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        print(f"Generating confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        
        ax1.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # 2. Metrics breakdown
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Normal', 'Attack']
        
        data = []
        for cls in classes:
            row = []
            for metric in metrics:
                row.append(report[cls][metric])
            data.append(row)
        
        # Create heatmap for metrics
        sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2,
                   xticklabels=[m.title() for m in metrics],
                   yticklabels=classes)
        
        ax2.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Class', fontsize=12)
        ax2.set_xlabel('Metric', fontsize=12)
        
        # Add overall metrics as text
        overall_text = f"""
        Overall Accuracy: {report['accuracy']:.3f}
        Macro Avg F1: {report['macro avg']['f1-score']:.3f}
        Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}
        """
        
        ax2.text(3.5, 0.5, overall_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        
        metrics_text = f"""
        Detection Rate: {tp/(tp+fn):.3f}
        False Alarm Rate: {fp/(fp+tn):.3f}
        True Negatives: {tn}
        False Positives: {fp}
        False Negatives: {fn}
        True Positives: {tp}
        """
        
        ax1.text(2.5, 0.5, metrics_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.suptitle(f'Model Performance Analysis - {model_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved to: {save_path}")
        
        # Print insights
        print(f"\n📊 CONFUSION MATRIX INSIGHTS for {model_name}:")
        print(f"   • Detection Rate (Recall): {tp/(tp+fn):.3f}")
        print(f"   • False Positive Rate: {fp/(fp+tn):.3f}")
        print(f"   • Overall Accuracy: {(tp+tn)/(tp+tn+fp+fn):.3f}")
        
        return save_path
    
    def plot_attack_patterns(self):
        """Visualize different attack patterns"""
        print("Generating attack patterns visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        attack_types = [
            ('DoS Attack', 'red', 'High packet rate, small packets'),
            ('MITM Attack', 'orange', 'Protocol anomalies, timing issues'),
            ('Data Injection', 'green', 'Anomalous data patterns'),
            ('Eavesdropping', 'purple', 'Unencrypted traffic, long sessions'),
            ('Replay Attack', 'blue', 'Duplicate packets, sequence issues'),
            ('Ransomware', 'darkred', 'Encryption patterns, ransom notes')
        ]
        
        for idx, (attack_name, color, description) in enumerate(attack_types):
            if idx >= 6:  # Only show first 6 attacks
                break
            
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            # Create radar-like plot for 4 features
            angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]
            
            # Generate attack-specific patterns
            np.random.seed(42 + idx)
            
            if attack_name == 'DoS Attack':
                values = [0.9, 0.8, 0.3, 0.7]  # High rate, small size, low duration, medium errors
                features = ['Packet Rate', 'Packet Size', 'Duration', 'Error Rate']
            elif attack_name == 'MITM Attack':
                values = [0.5, 0.7, 0.8, 0.6]  # Medium rate, large size, long duration, medium errors
                features = ['Protocol Anomaly', 'Packet Size', 'Duration', 'Entropy']
            elif attack_name == 'Data Injection':
                values = [0.6, 0.9, 0.4, 0.8]  # Medium rate, very large size, short duration, high errors
                features = ['Payload Size', 'Data Anomaly', 'Duration', 'CRC Errors']
            elif attack_name == 'Eavesdropping':
                values = [0.7, 0.5, 0.9, 0.3]  # High rate, medium size, very long duration, low errors
                features = ['Session Rate', 'Packet Size', 'Duration', 'Encryption']
            elif attack_name == 'Replay Attack':
                values = [0.4, 0.6, 0.7, 0.5]  # Low rate, medium size, medium duration, medium errors
                features = ['Duplicate Rate', 'Sequence Gap', 'Timing', 'Authentication']
            else:  # Ransomware
                values = [0.8, 0.7, 0.6, 0.9]  # High rate, large size, medium duration, high errors
                features = ['Encryption Rate', 'File Access', 'System Calls', 'Network Traffic']
            
            values = [v + np.random.uniform(-0.1, 0.1) for v in values]
            values = [max(0.1, min(0.95, v)) for v in values]  # Clip values
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=8)
            ax.fill(angles, values, alpha=0.25, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(features, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(attack_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add description
            ax.text(0.5, -0.25, description[:40] + '...' if len(description) > 40 else description,
                   ha='center', va='center', transform=ax.transAxes, fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle('IoT Attack Pattern Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/attack_patterns.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Attack patterns visualization saved to: {save_path}")
        return save_path
    
    def plot_iot_architecture(self):
        """Visualize IoT architecture layers"""
        print("Generating IoT architecture visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        layers = [
            {
                'name': 'Perception Layer',
                'color': 'lightblue',
                'components': ['Sensors', 'Actuators', 'RFID', 'Cameras', 'GPS'],
                'icon': '📡'
            },
            {
                'name': 'Network Layer',
                'color': 'lightgreen',
                'components': ['Gateways', 'Routers', 'Wi-Fi AP', '5G/LTE', 'Protocols'],
                'icon': '🌐'
            },
            {
                'name': 'Application Layer',
                'color': 'lightcoral',
                'components': ['Cloud Platform', 'Mobile Apps', 'Web Dashboard', 'Analytics', 'AI'],
                'icon': '📱'
            }
        ]
        
        for idx, (ax, layer) in enumerate(zip(axes, layers)):
            # Draw layer box
            rect = plt.Rectangle((0.1, 0.1), 0.8, 0.7, 
                                facecolor=layer['color'], alpha=0.3,
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add layer title
            ax.text(0.5, 0.85, f"{layer['icon']} {layer['name']}", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold')
            
            # Add components
            for i, component in enumerate(layer['components']):
                y_pos = 0.75 - i * 0.12
                ax.text(0.5, y_pos, f"• {component}", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="white", alpha=0.8))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Add connecting arrows
        fig.text(0.32, 0.5, '→', fontsize=30, ha='center', color='gray')
        fig.text(0.66, 0.5, '→', fontsize=30, ha='center', color='gray')
        
        plt.suptitle('IoT System Architecture', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/iot_architecture.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ IoT architecture visualization saved to: {save_path}")
        return save_path
    
    def plot_performance_over_time(self):
        """Plot model performance improvement over training"""
        print("Generating performance over time visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Simulate training progress
        epochs = list(range(1, 21))
        
        # Model 1: Random Forest
        rf_accuracy = [0.75 + 0.02*i + np.random.uniform(-0.03, 0.03) for i in range(20)]
        rf_accuracy = [min(0.98, max(0.75, x)) for x in rf_accuracy]
        
        # Model 2: Neural Network
        nn_accuracy = [0.70 + 0.025*i + np.random.uniform(-0.05, 0.05) for i in range(20)]
        nn_accuracy = [min(0.97, max(0.70, x)) for x in nn_accuracy]
        
        # Model 3: Ensemble
        ensemble_accuracy = [0.78 + 0.015*i + np.random.uniform(-0.02, 0.02) for i in range(20)]
        ensemble_accuracy = [min(0.99, max(0.78, x)) for x in ensemble_accuracy]
        
        # Plot accuracy over epochs
        ax1.plot(epochs, rf_accuracy, 'b-o', linewidth=2, markersize=6, label='Random Forest')
        ax1.plot(epochs, nn_accuracy, 'r-s', linewidth=2, markersize=6, label='Neural Network')
        ax1.plot(epochs, ensemble_accuracy, 'g-^', linewidth=2, markersize=6, label='Ensemble')
        
        ax1.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Model Accuracy Over Training', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.7, 1.0)
        
        # Plot loss over epochs
        rf_loss = [0.5 * (0.9**i) + np.random.uniform(0, 0.05) for i in range(20)]
        nn_loss = [0.7 * (0.85**i) + np.random.uniform(0, 0.08) for i in range(20)]
        
        ax2.plot(epochs, rf_loss, 'b-o', linewidth=2, markersize=6, label='Random Forest')
        ax2.plot(epochs, nn_loss, 'r-s', linewidth=2, markersize=6, label='Neural Network')
        
        ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.8)
        
        plt.suptitle('Model Training Progress', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = f"{self.output_dir}/training_progress.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training progress visualization saved to: {save_path}")
        return save_path

# Test the visualizer
if __name__ == "__main__":
    print("Testing EnhancedVisualizer...")
    
    visualizer = EnhancedVisualizer()
    
    # Test with sample data
    np.random.seed(42)
    
    # Create sample model results
    sample_results = {
        'Random Forest': {
            'accuracy': 0.965,
            'precision': 0.954,
            'recall': 0.938,
            'f1_score': 0.946,
            'train_time': 2.3
        },
        'XGBoost': {
            'accuracy': 0.972,
            'precision': 0.961,
            'recall': 0.958,
            'f1_score': 0.959,
            'train_time': 3.1
        },
        'Neural Network': {
            'accuracy': 0.975,
            'precision': 0.968,
            'recall': 0.965,
            'f1_score': 0.966,
            'train_time': 45.2
        },
        'Ensemble': {
            'accuracy': 0.981,
            'precision': 0.973,
            'recall': 0.975,
            'f1_score': 0.974,
            'train_time': 12.4
        }
    }
    
    # Generate visualizations
    visualizer.plot_model_comparison(sample_results)
    visualizer.plot_attack_patterns()
    visualizer.plot_iot_architecture()
    visualizer.plot_performance_over_time()
    
    print("\n✅ EnhancedVisualizer test complete!")
    print(f"Check '{visualizer.output_dir}' for generated graphs")