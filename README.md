# AutoML Web Dashboard

This Streamlit AutoML Web Dashboard enables users to upload datasets, automate data preprocessing, and train/evaluate various ML models. It features interactive performance charts, actionable BI insights, model export capabilities, and supply chain analytics for demand forecasting and inventory risk assessment.

## Quick Start: Copy & Paste in Terminal

Run the following commands in your terminal to clone the repository, install dependencies, and start the application:

```bash
# Clone the repository
git clone https://github.com/Akshay-Notfound/AutoML-Web-Dashboard.git

# Navigate to the project directory
cd AutoML-Web-Dashboard

# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

## Features Complete Guide

- **Data Upload**: Supports CSV and Excel file formats.
- **Data Preprocessing**: Handling missing values, categorical encoding, feature scaling, and train-test splits.
- **Model Training & Evaluation**: Train Logistic Regression, Decision Tree, Random Forest, KNN, and SVM models. Compare performance using Accuracy, Precision, Recall, and F1 Score with interactive charts.
- **Business Intelligence (BI) Insights**: Extract actionable insights based on feature importance and model reliability. Export the best model for future use.
- **Supply Chain Analytics**: Built-in demand forecasting (Moving Average & Exponential Smoothing) and inventory volatility & risk analysis matrix.

## Architecture

This architecture diagram illustrates the flow of data and interaction between the underlying modules of the Streamlit application.

```mermaid
graph TD
    %% Styling
    classDef ui fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:white;
    classDef logic fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:white;
    classDef data fill:#FF9800,stroke:#E65100,stroke-width:2px,color:white;
    classDef model fill:#9C27B0,stroke:#4A148C,stroke-width:2px,color:white;
    classDef user fill:#607D8B,stroke:#37474F,stroke-width:2px,color:white;

    User([User]):::user --> |Uploads CSV/Excel| appUI[Streamlit Core UI]:::ui
    appUI --> Sidebar[Navigation Sidebar]:::ui
    
    %% Session State (Data Storage)
    subgraph DataState [Streamlit Session State]
        raw[(Raw Dataset)]:::data
        clean[(Cleaned Data)]:::data
        split[(Train/Test Split)]:::data
        models[(Trained Pipeline & Models)]:::data
    end

    appUI --> |Saves to State| raw

    %% Core Modules
    subgraph Modules [Application Modules]
        Preprocess[Data Preprocessing]:::logic
        Train[Model Training & Evaluation]:::logic
        BI[Business Intelligence]:::logic
        SupplyChain[Supply Chain Analytics]:::logic
    end

    %% Preprocessing Flow
    Sidebar --> |Selects Step 2| Preprocess
    raw -.-> Preprocess
    Preprocess --> |Imputation, Encoding, Scaling| clean
    Preprocess --> |Train/Test Split| split

    %% Model Evaluation Flow
    Sidebar --> |Selects Step 3| Train
    split -.-> Train
    Train --> |Scikit-learn: LR, DT, RF, KNN, SVM| ML[ML Engines]:::model
    ML --> |Output: Acc, F1, ROC, Radar Charts| models

    %% BI Insights Flow
    Sidebar --> |Selects Step 4| BI
    models -.-> BI
    BI --> |Joblib| Export[Export Model .pkl]:::logic
    BI --> Inference[Real-time Inference]:::logic
    BI --> Insights[Extract Feature Importances]:::logic

    %% Supply Chain Flow
    Sidebar --> |Selects Step 5| SupplyChain
    clean -.-> SupplyChain
    SupplyChain --> |SMA / EMA Forecast| Forecast[Demand Forecasting]:::model
    SupplyChain --> |Grouped Volatility| Risk[Inventory Risk Matrix]:::model

    %% Connect back to User Interface Elements
    Export -.-> |Download| User
    Inference -.-> |Predictions| User
    Insights -.-> |Display| appUI
    Forecast -.-> |Plotly Vis| appUI
    Risk -.-> |Plotly Vis| appUI
    ML -.-> |Plotly Vis| appUI
```
