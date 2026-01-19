# NHL-xG-Prediction-System

A **full-stack machine learning system** for predicting **NHL Expected Goals (xG)** in real time from play-by-play data.  
The project covers the complete lifecycle of a production-oriented ML system, from data ingestion and modeling to containerized deployment and live visualization.



## Project Overview

Expected Goals (xG) is a widely used hockey analytics metric that estimates the probability that a shot results in a goal based on contextual features such as shot location and angle.

This project implements:
- an **xG prediction model** trained on NHL play-by-play data,
- a **model-as-a-service** architecture exposed through a REST API,
- a **real-time client** that processes live (or historical) game events,
- and an **interactive dashboard** for visualizing predictions and game dynamics.

The system is designed to be modular, reproducible, and easily extensible to new models or features.



## System Architecture

The project follows a service-oriented architecture:

1. **Data Client**
   - Retrieves NHL play-by-play data from the official NHL API
   - Extracts and preprocesses shot-level features compatible with the model

2. **Prediction Service (Flask)**
   - Serves trained xG models via a REST API
   - Supports dynamic model loading and hot-swapping
   - Logs predictions and system events for debugging and inspection

3. **Visualization Client (Streamlit)**
   - Displays live or replayed game events
   - Visualizes xG predictions and game statistics in real time
   - Communicates with the prediction service through HTTP requests

4. **Containerization (Docker)**
   - Ensures reproducibility and environment consistency
   - Separates concerns between inference service and visualization layer



## Machine Learning Pipeline

- **Task**: Binary classification (goal vs. no goal)
- **Target**: Probability of a shot resulting in a goal (xG)
- **Features**:
  - Shot distance
  - Shot angle
  - Empty-net indicator
- **Models**:
  - Logistic Regression (baseline, interpretable)
- **Evaluation**:
  - ROC curves and AUC
  - Calibration and probability-based assessment

The predicted probabilities are used directly as xG values rather than hard class labels.



## Repository Structure

```text
NHL-xG-Prediction-System/
│
├── ift6758/                  # Core package
│   ├── client/               # Data and serving clients
│   ├── data/                 # Data loading and preprocessing
│   ├── serving/              # Flask prediction service
│   └── utils/                # Shared utilities
│
├── streamlit_app/            # Interactive dashboard
│
├── docker/                   # Dockerfiles and container configs
│
├── notebooks/                # Exploratory analysis and prototyping
│
├── requirements.txt
├── README.md
└── docker-compose.yml
```



## Running the Project

### Prerequisites
- Python 3.9+
- Docker

### Build and run services
```bash
docker-compose up --build
```

### Access the applications

- URL for MAC/Linux : http://0.0.0.0:8501
- URL for Windows : http://localhost:8501



## Key Technologies

- Python 
- scikit-learn 
- Flask (REST API)
- Streamlit (interactive visualization)
- Docker (containerization)
- Weights & Biases (experiment tracking & model registry)
- Learning Outcomes 
- This project demonstrates:
- End-to-end ML system design 
- Feature engineering from real-world, noisy data 
- Model deployment as a service 
- API-based ML inference 
- Real-time data processing and visualization 
- Reproducible and modular ML engineering practices 





Add advanced feature engineering (rebound shots, game state, prior events)

Integrate more expressive models (Gradient Boosting, Neural Networks)
