import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, style='seaborn'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.output_dir = "outputs"
        
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_iot_architecture(self, save_path=None):
        """
        Create IoT architecture visualization
        """
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Layer 1: Perception Layer
        ax[0].text(0.5, 0.9, 'Perception Layer', ha='center', va='center', 
                  fontsize=14, fontweight='bold', transform=ax[0].transAxes)
        ax[0].add_patch(Rectangle((0.1, 0.1), 0.8, 0.7, fill=False, edgecolor='blue', lw=2))
        
        # Add sensor icons
        sensors = ['Temperature', 'Motion', 'Camera', 'GPS']
        for i, sensor in enumerate(sensors):
            y_pos = 0.7 - i * 0.15
            ax[0].text(0.5, y_pos, sensor, ha='center', va='center', 
                      bbox=dict(boxstyle="circle,pad=0.3", facecolor="lightblue"))
        
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].axis('off')
        ax[0].set_title('Sensors & Actuators', fontsize=12, fontweight='bold')
        
        # Layer 2: Network Layer
        ax[1].text(0.5, 0.9, 'Network Layer', ha='center', va='center', 
                  fontsize=14, fontweight='bold', transform=ax[1].transAxes)
        ax[1].add_patch(Rectangle((0.1, 0.1), 0.8, 0.7, fill=False, edgecolor='green', lw=2))
        
        # Network components
        components = ['Gateway', 'Router', 'Switch', 'AP']
        for i, comp in enumerate(components):
            y_pos = 0.7 - i * 0.15
            ax[1].text(0.5, y_pos, comp, ha='center', va='center', 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].axis('off')
        ax[1].set_title('Communication & Routing', fontsize=12, fontweight='bold')
        
        # Layer 3: Application Layer
        ax[2].text(0.5, 0.9, 'Application Layer', ha='center', va='center', 
                  fontsize=14, fontweight='bold', transform=ax[2].transAxes)
        ax[2].add_patch(Rectangle((0.1, 0.1), 0.8, 0.7, fill=False, edgecolor='red', lw=2))
        
        # Applications
        apps = ['Cloud Server', 'Mobile App', 'Web Dashboard', 'Database']
        for i, app in enumerate(apps):
            y_pos = 0.7 - i * 0.15
            ax[2].text(0.5, y_pos, app, ha='center', va='center', 
                      bbox=dict(boxstyle="round4,pad=0.3", facecolor="lightcoral"))
        
        ax[2].set_xlim(0, 1)
        ax[2].set_ylim(0, 1)
        ax[2].axis('off')
        ax[2].set_title('Services & Applications', fontsize=12, fontweight='bold')
        
        # Add connecting arrows
        fig.text(0.32, 0.5, '→', fontsize=20, ha='center')
        fig.text(0.66, 0.5, '→', fontsize=20, ha='center')
        
        plt.suptitle('IoT System Architecture', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/iot_architecture.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
        plt.show()
    
    def plot_attack_distribution(self, y, save_path=None):
        """
        Plot attack vs normal distribution
        """
        attack_count = sum(y == 1)
        normal_count = sum(y == 0)
        
        labels = ['Normal Traffic', 'Attack Traffic']
        sizes = [normal_count, attack_count]
        colors = ['#66b3ff', '#ff6666']
        explode = (0.1, 0)  # explode the attack slice
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.axis('equal')
        ax1.set_title('Traffic Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Count by Class', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value}', ha='center', va='bottom')
        
        plt.suptitle('Attack vs Normal Traffic Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/attack_distribution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack distribution plot saved to {save_path}")
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=15, save_path=None):
        """
        Plot feature importance for tree-based models
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model doesn't have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_features))
        
        bars = ax.barh(y_pos, top_importances, align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features for Intrusion Detection', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_importances)):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/feature_importance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history for neural networks
        """
        if history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/training_history.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.show()
    
    def plot_attack_patterns(self, normal_features, attack_features, save_path=None):
        """
        Visualize differences between normal and attack patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Select key features to visualize
        features_to_plot = ['Packet Size', 'Packet Count', 'Error Rate', 'Duration']
        
        # Create sample data if not provided
        if normal_features is None or attack_features is None:
            np.random.seed(42)
            normal_features = {
                'Packet Size': np.random.normal(500, 150, 100),
                'Packet Count': np.random.poisson(5, 100),
                'Error Rate': np.random.beta(1, 9, 100),
                'Duration': np.random.exponential(1.0, 100)
            }
            
            attack_features = {
                'Packet Size': np.random.normal(100, 50, 100),
                'Packet Count': np.random.poisson(50, 100),
                'Error Rate': np.random.beta(8, 2, 100),
                'Duration': np.random.exponential(0.1, 100)
            }
        
        # Plot each feature comparison
        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # Plot distributions
            ax.hist(normal_features[feature], alpha=0.5, label='Normal', bins=20, color='blue')
            ax.hist(attack_features[feature], alpha=0.5, label='Attack', bins=20, color='red')
            
            ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Normal vs Attack Traffic Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/attack_patterns.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack patterns plot saved to {save_path}")
        plt.show()
    
    def create_interactive_dashboard(self, metrics_dict, save_path=None):
        """
        Create an interactive dashboard using Plotly
        """
        if save_path is None:
            save_path = f"{self.output_dir}/interactive_dashboard.html"
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Metrics', 'Confusion Matrix',
                           'Feature Importance', 'ROC Curve'),
            specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Performance Metrics Bar Chart
        metrics_names = list(metrics_dict.keys())[:4]  # First 4 metrics
        metrics_values = list(metrics_dict.values())[:4]
        
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values,
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                  text=[f'{v:.3f}' for v in metrics_values],
                  textposition='auto'),
            row=1, col=1
        )
        
        # 2. Confusion Matrix Heatmap (dummy data)
        confusion_data = [[50, 5], [3, 42]]
        fig.add_trace(
            go.Heatmap(z=confusion_data,
                      x=['Predicted Normal', 'Predicted Attack'],
                      y=['Actual Normal', 'Actual Attack'],
                      colorscale='Blues',
                      showscale=True,
                      text=confusion_data,
                      texttemplate="%{text}",
                      textfont={"size": 16}),
            row=1, col=2
        )
        
        # 3. Feature Importance (dummy data)
        features = ['Packet Size', 'Duration', 'Src Bytes', 'Dst Bytes', 'Count']
        importance = [0.25, 0.18, 0.15, 0.12, 0.10]
        
        fig.add_trace(
            go.Bar(x=importance, y=features, orientation='h',
                  marker_color='lightseagreen'),
            row=2, col=1
        )
        
        # 4. ROC Curve (dummy data)
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Dummy ROC curve
        
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines',
                      line=dict(color='darkorange', width=3),
                      name='ROC Curve'),
            row=2, col=2
        )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(color='navy', width=2, dash='dash'),
                      name='Random Classifier'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="AI-Based IDS Dashboard",
            title_font=dict(size=24, family="Arial", color="darkblue")
        )
        
        # Save interactive plot
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
        
        return fig