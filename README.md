# AI-Based Intrusion Detection System for IoT Networks

## Project Overview
This project implements an AI-based Intrusion Detection System (IDS) specifically designed for Internet of Things (IoT) networks. The system uses machine learning and deep learning techniques to detect malicious activities in IoT environments.

## Features
- **Multi-Layer IoT Architecture Modeling**
- **Simulated Attack Generation** (DoS, MITM, Eavesdropping, Data Injection)
- **Multiple AI Models** (Random Forest, CNN, LSTM, SVM, Ensemble)
- **Comprehensive Evaluation Metrics**
- **Interactive Visualizations**
- **Real-time Detection Simulation**

## Installation

### Prerequisites
- Python 3.8+
- Kali Linux (recommended) or any Linux distribution
- 4GB RAM minimum, 8GB recommended

### Setup on Kali Linux
```bash
# Clone the repository
git clone https://github.com/yourusername/iot-ids-project.git
cd iot-ids-project

# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt