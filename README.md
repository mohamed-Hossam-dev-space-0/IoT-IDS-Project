# AI-Based Intrusion Detection System for IoT Networks

An AI-powered Intrusion Detection System designed to detect and prevent security threats in IoT networks using machine learning.

## Features

- Multiple AI models (Random Forest, XGBoost, Neural Networks)
- Real-time attack detection
- Live monitoring dashboard
- Attack simulation laboratory
- 3D IoT architecture visualization

## Project Structure

```
Project/
│
├── 📄 main.py                      # Main entry point
├── 📄 compare_models.py            # AI models comparison
├── 📄 run_dashboard.py             # Live dashboard
├── 📄 simulate_attacks_real.py     # Attack simulation
├── 📄 requirements.txt             # Required libraries
├── 📄 config.yaml                  # Project configuration
│
├── 📂 data/                        # Data files
│   ├── data_loader_enhanced.py    # Data generation
│   ├── attack_simulator.py        # Attack simulator
│   └── dataset_downloader.py      # Real data downloader
│
├── 📂 models/                      # AI models
│   ├── model_factory.py           # Model factory
│   └── neural_models.py           # Deep learning models
│
├── 📂 utils/                       # Utility tools
│   ├── visualizer_enhanced.py     # Charts and visualizations
│   └── iot_architecture_3d.py     # 3D architecture visualization
│
└── 📂 outputs/                     # Outputs
    ├── graphs/                     # Charts

## Installation

```bash
# Create project directory
mkdir IoT-IDS-Enhanced
cd IoT-IDS-Enhanced

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# Optional: For dashboard and enhanced features
pip install dash plotly xgboost
```

## Quick Start

```bash
# Run demo
python main.py --mode demo

# Analyze architecture
python main.py --mode analyze

# Simulate attacks
python simulate_attacks_real.py

# Compare models
python compare_models.py

# Run dashboard (optional)
python run_dashboard.py
```

## Attack Types Detected

- **DoS (Denial of Service)** - Flooding attacks
- **MITM (Man-in-the-Middle)** - Communication interception
- **Data Injection** - Malicious data insertion
- **Eavesdropping** - Unauthorized monitoring

## Model Performance

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 96.8% | 2.3s |
| XGBoost | 97.2% | 3.1s |
| Neural Network | 97.5% | 45.2s |
| Ensemble | 98.1% | 12.4s |

## Usage Examples

### Basic Detection

```python
from data.data_loader_enhanced import EnhancedDataLoader
from models.model_factory import ModelFactory

# Load data
loader = EnhancedDataLoader()
X_train, X_test, y_train, y_test = loader.load_enhanced_dataset()

# Train model
factory = ModelFactory()
model = factory.create_model('random_forest')
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Run Simulations

```bash
python simulate_attacks_real.py

# Select from menu:
# 1. Normal IoT traffic
# 2. DoS attack
# 3. MITM attack
# 4. Data injection
# 5. Comprehensive simulation
```

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  samples: 10000
  attack_ratio: 0.2
  
models:
  random_forest:
    n_estimators: 100
    max_depth: 15
```

## Outputs

All results are saved in the `outputs/` directory:

- `outputs/graphs/` - Visualizations

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Optional:
- Dash (for live dashboard)
- Plotly (for interactive plots)
- XGBoost (for enhanced models)

## Troubleshooting

**Import errors:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

**Dashboard won't start:**
```bash
pip install dash plotly
```

**Memory issues:**
- Reduce dataset size in `config.yaml`
- Use smaller models

## Academic Use

This project is part of the Cyber Physical Systems Security (CCY4301) course at the Arab Academy for Science, Technology & Maritime Transport.

## License

Educational use only - Not for commercial distribution.

## Team

- Karem Osama
- Mohamed Hossam Mohamed
- Adham Hamada 
- Mohamed Tamer 
- Omar Hesham Hamed
