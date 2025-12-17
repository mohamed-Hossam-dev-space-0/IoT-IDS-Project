#!/usr/bin/env python3
"""
Model Comparison Utility for IoT-IDS
Run with: python compare_models.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 COMPREHENSIVE AI MODEL COMPARISON FOR IOT-IDS")
print("="*70)

# Try to import our enhanced modules
try:
    from data.data_loader_enhanced import EnhancedDataLoader
    from models.model_factory import ModelFactory
    from utils.visualizer_enhanced import EnhancedVisualizer
    
    print("✅ Enhanced modules loaded successfully!")
    
except ImportError as e:
    print(f"⚠️  Could not load enhanced modules: {e}")
    print("Using simplified comparison...")
    
    # Define fallback classes
    class SimpleDataLoader:
        def load_data(self, n_samples=1000):
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
    
    class SimpleModelFactory:
        def create_model(self, model_type):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            
            if model_type == 'random_forest':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'svm':
                return SVC(probability=True, random_state=42)
            elif model_type == 'mlp':
                return MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
    
    EnhancedDataLoader = SimpleDataLoader
    ModelFactory = SimpleModelFactory
    EnhancedVisualizer = None

def compare_all_models():
    """Compare all available AI models for IoT intrusion detection"""
    
    # Initialize components
    data_loader = EnhancedDataLoader()
    
    print("\n📂 Loading dataset...")
    
    try:
        # Try to load enhanced dataset
        X, y, feature_names = data_loader.load_enhanced_dataset(n_samples=2000)
        print(f"✅ Enhanced dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    except:
        # Fallback to simple dataset
        X, y, feature_names = data_loader.load_data(n_samples=1000)
        print(f"✅ Simple dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Testing samples: {X_test.shape[0]}")
    print(f"   • Features: {X_train.shape[1]}")
    print(f"   • Normal samples: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    print(f"   • Attack samples: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    
    # Define models to compare
    models_to_compare = [
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('svm', 'Support Vector Machine'),
        ('mlp', 'Multi-Layer Perceptron'),
        ('cnn', 'Convolutional Neural Network'),
        ('lstm', 'Long Short-Term Memory')
    ]
    
    # Check if XGBoost is available
    try:
        import xgboost
        print("✅ XGBoost available for comparison")
    except ImportError:
        print("⚠️  XGBoost not installed. Removing from comparison.")
        models_to_compare = [m for m in models_to_compare if m[0] != 'xgboost']
    
    results = {}
    training_times = {}
    
    print("\n" + "="*70)
    print("🏋️  TRAINING AND EVALUATING MODELS")
    print("="*70)
    
    for model_key, model_name in models_to_compare:
        print(f"\n▶️  {model_name}:")
        print(f"   {'─' * (len(model_name) + 2)}")
        
        try:
            # Create and train model
            factory = ModelFactory()
            
            if hasattr(factory, 'create_model'):
                model_obj = factory.create_model(model_key)
            else:
                model_obj = factory.create_model(model_key)
            
            # Train with timing
            train_start = time.time()
            
            if hasattr(model_obj, 'train'):
                # Our enhanced model interface
                model_obj.train(X_train, y_train)
                train_time = time.time() - train_start
                
                # Evaluate
                if hasattr(model_obj, 'evaluate'):
                    metrics = model_obj.evaluate(X_test, y_test)
                else:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_pred = model_obj.predict(X_test)
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, zero_division=0)
                    }
            else:
                # Standard sklearn interface
                model_obj.fit(X_train, y_train)
                train_time = time.time() - train_start
                
                # Evaluate
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_pred = model_obj.predict(X_test)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
            
            training_times[model_name] = train_time
            results[model_name] = metrics
            
            print(f"   Training time: {train_time:.2f}s")
            print(f"   📈 Accuracy:    {metrics['accuracy']*100:.2f}%")
            print(f"   🎯 Precision:   {metrics['precision']*100:.2f}%")
            print(f"   🔍 Recall:      {metrics['recall']*100:.2f}%")
            print(f"   ⚖️  F1-Score:    {metrics['f1_score']*100:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Calculate ensemble performance (weighted average of best models)
    print("\n" + "="*70)
    print("🏆 CALCULATING ENSEMBLE PERFORMANCE")
    print("="*70)
    
    if results:
        # Sort models by F1-Score
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        print(f"\n   Top models for ensemble calculation:")
        for i, (model_name, metrics) in enumerate(sorted_models[:3]):
            print(f"      {i+1}. {model_name}: F1={metrics['f1_score']:.3f}")
        
        # Calculate weighted average for ensemble
        ensemble_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            weighted_sum = 0
            total_weight = 0
            
            for i, (model_name, metrics) in enumerate(sorted_models[:3]):
                weight = 0.4 if i == 0 else 0.3 if i == 1 else 0.3
                weighted_sum += metrics[metric] * weight
                total_weight += weight
            
            ensemble_metrics[metric] = weighted_sum / total_weight
        
        # Add average training time
        if training_times:
            ensemble_metrics['train_time'] = np.mean(list(training_times.values()))
        
        results['Voting Ensemble'] = ensemble_metrics
        
        print(f"\n   🎯 Estimated Ensemble Performance:")
        print(f"      • Accuracy:    {ensemble_metrics['accuracy']*100:.2f}%")
        print(f"      • Precision:   {ensemble_metrics['precision']*100:.2f}%")
        print(f"      • Recall:      {ensemble_metrics['recall']*100:.2f}%")
        print(f"      • F1-Score:    {ensemble_metrics['f1_score']*100:.2f}%")
    
    # Generate comprehensive comparison report
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE COMPARISON REPORT")
    print("="*70)
    
    if results:
        # Create visualizations if available
        if EnhancedVisualizer:
            try:
                visualizer = EnhancedVisualizer()
                visualizer.plot_model_comparison(results)
            except:
                print("⚠️  Could not generate enhanced visualizations")
        
      # Create simple visualization
        print("\n📈 Creating comparison chart...")
        
        # SIMPLIFIED FIX: Use a single figure with 2 subplots, properly spaced
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Model comparison bar chart (Accuracy & F1-Score only)
        models = list(results.keys())
        accuracy_values = [results[m].get('accuracy', 0) for m in models]
        f1_values = [results[m].get('f1_score', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        # Create bars with proper spacing
        bars1 = ax1.bar(x - width/2, accuracy_values, width, 
                       label='Accuracy',
                       color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, f1_values, width,
                       label='F1-Score',
                       color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs F1-Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        
        # Truncate model names for display
        display_names = []
        for m in models:
            if len(m) > 12:
                display_names.append(m[:10] + '...')
            else:
                display_names.append(m)
        
        ax1.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars (SMALLER font to avoid overlap)
        for bars, color in [(bars1, '#3498db'), (bars2, '#2ecc71')]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label if value > 0
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', 
                            fontsize=8, fontweight='bold', color=color)
        
        # 2. Training time comparison - FIXED SPACING
        if training_times:
            model_names = list(training_times.keys())
            times = list(training_times.values())
            
            # Use the SAME order as first chart for consistency
            sorted_indices = np.argsort([models.index(m) for m in model_names])
            sorted_names = [model_names[i] for i in sorted_indices]
            sorted_times = [times[i] for i in sorted_indices]
            
            x_time = np.arange(len(sorted_names))
            bars = ax2.bar(x_time, sorted_times, color='lightcoral', alpha=0.7, 
                          edgecolor='darkred', linewidth=1, width=0.6)
            
            ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_time)
            
            # Use same truncated names
            time_display_names = []
            for m in sorted_names:
                if len(m) > 12:
                    time_display_names.append(m[:10] + '...')
                else:
                    time_display_names.append(m)
            
            ax2.set_xticklabels(time_display_names, rotation=45, ha='right', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Auto-adjust y-axis for time chart
            max_time = max(sorted_times) if sorted_times else 1
            ax2.set_ylim(0, max_time * 1.2)
            
            # Add time labels
            for bar, time_val in zip(bars, sorted_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max_time * 0.02,
                        f'{time_val:.2f}s', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold', color='darkred')
        else:
            ax2.text(0.5, 0.5, 'No training time data\navailable',
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=12, fontweight='bold')
            ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        
        # Add space between subplots
        plt.suptitle('AI Model Comparison for IoT Intrusion Detection', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # CRITICAL: Add proper spacing between plots
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        # Save figure
        import os
        os.makedirs("outputs/graphs", exist_ok=True)
        save_path = "outputs/graphs/model_comparison_clean.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.show()
        
        print(f"✅ Comparison chart saved to: {save_path}")
        
        # Print ranking table
        print("\n📑 MODEL RANKING (by F1-Score):")
        print("-" * 85)
        print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
        print("-" * 85)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            
            # Handle training time safely
            try:
                time_val = training_times.get(model_name, 0)
                if isinstance(time_val, str):
                    if time_val == 'N/A' or time_val == '':
                        time_str = "N/A"
                    else:
                        time_str = f"{float(time_val):.2f}"
                else:
                    time_str = f"{float(time_val):.2f}"
            except (ValueError, TypeError):
                time_str = "N/A"
            
            print(f"{rank:<6} {model_name:<25} {acc:<10.3f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f} {time_str:<10}")
        
        print("-" * 85)
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS:")
        print("-" * 60)
        
        best_model = sorted_results[0][0]
        best_f1 = sorted_results[0][1].get('f1_score', 0)
        
        print(f"   • Best Model: {best_model} (F1-Score: {best_f1:.3f})")
        
        if 'Voting Ensemble' in results:
            print(f"   • For maximum accuracy: Use Voting Ensemble")
            print(f"   • For production: Consider computational requirements")
        elif 'Random Forest' in best_model:
            print(f"   • Good choice for IoT: Fast training, good accuracy")
            print(f"   • Easy to interpret and debug")
        elif 'XGBoost' in best_model:
            print(f"   • Excellent accuracy, good for production")
            print(f"   • Requires more tuning but worth it")
        elif 'Neural Network' in best_model:
            print(f"   • Best for complex patterns")
            print(f"   • Requires more data and computational resources")
        
        print(f"\n💡 SELECTION CRITERIA:")
        print("-" * 60)
        print("   • High accuracy: Choose top F1-Score model")
        print("   • Fast training: Check training times")
        print("   • IoT constraints: Consider model size and speed")
        print("   • Interpretability: Random Forest is most interpretable")
    
    else:
        print("❌ No results to compare!")
    
    print("\n" + "="*70)
    print("✅ MODEL COMPARISON COMPLETE")
    print("="*70)
    
    
    return results

if __name__ == "__main__":
    results = compare_all_models()