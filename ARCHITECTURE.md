# AutoML Web Dashboard Architecture

This architecture diagram illustrates the flow of data and interaction between the underlying modules of your Streamlit application. 

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

### Component Details
- **User Interface Layer**: Built completely in Streamlit, driven by a Sidebar navigation system.
- **Session State Management**: Serves as the database/memory for the app, storing uploaded raw data, cleaned features, training splits, and model artifacts as the user progresses through different pages.
- **Data Preprocessing**: Handles missing values, performs one-hot/label encoding, applies MinMax/Standard scaling, and manages the Train/Test partitioning.
- **Model Training**: Incorporates Scikit-Learn classifiers. Generates robust classification metrics, confusion matrices, and ROC/PR charts using Plotly.
- **Business Intelligence**: Takes the best performing model from the training phase to deduce feature impact. Also supports real-time new data inferences and model downloads using `joblib`.
- **Supply Chain Analytics**: An independent module that uses the cleaned data to calculate historical Moving Averages (SMA/EMA) and inventory volatility maps for demand analysis.
