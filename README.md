# AI-Based Intrusion Detection System for IoT Networks

## 🎯 Project Overview
An enhanced AI-based Intrusion Detection System (IDS) specifically designed for IoT networks. This system uses multiple machine learning models to detect various types of attacks in real-time.

## 📁 Project Structure
IoT-IDS-Enhanced/
├── main.py # Main control panel
├── run_dashboard.py # Live monitoring dashboard
├── simulate_attacks_real.py # Attack simulation laboratory
├── compare_models.py # Model comparison utility
├── requirements.txt # Python dependencies
├── config.yaml # Configuration file
│
├── data/ # Data handling
│ ├── data_loader_enhanced.py # Enhanced dataset generation
│ └── [other data modules]
│
├── models/ # AI models
│ ├── model_factory.py # Model creation factory
│ └── [other model modules]
│
├── utils/ # Utilities
│ ├── visualizer_enhanced.py # Professional visualizations
│ ├── iot_architecture_3d.py # 3D architecture visualization
│ └── [other utility modules]
│
├── outputs/ # Generated outputs
│ ├── graphs/ # Visualization graphs
│ ├── reports/ # Generated reports
│ ├── models/ # Saved models
│ └── simulations/ # Simulation results
│
└── README.md # This file


## 🚀 Quick Start

### 1. Installation
```bash
# Clone or create project directory
mkdir IoT-IDS-Enhanced
cd IoT-IDS-Enhanced

# Create virtual environment (recommended)
python3 -m venv iot-env
source iot-env/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn