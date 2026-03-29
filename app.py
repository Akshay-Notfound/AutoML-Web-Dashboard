import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page configuration for a wider layout
st.set_page_config(
    page_title="AutoML Web Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Real-Time Dashboard Custom CSS
st.markdown("""
<style>
    /* Main App Background - Sleek Dark Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Elegant Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #58a6ff;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(88, 166, 255, 0.15);
        border-color: #58a6ff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }
    
    /* Container Styling */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #30363d;
        overflow: hidden;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #238636;
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(240, 246, 252, 0.1);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: rgba(240, 246, 252, 0.2);
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.4);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f6fc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Alert / Info boxes */
    .stAlert {
        background-color: #1f2428;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #c9d1d9;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff;
    }

</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables to persist data across pages
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'clean_data' not in st.session_state:
    st.session_state['clean_data'] = None
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = {}
if 'best_model_name' not in st.session_state:
    st.session_state['best_model_name'] = None
if 'target_col' not in st.session_state:
    st.session_state['target_col'] = None


def render_sidebar():
    """Renders the sidebar navigation and returns the selected page."""
    st.sidebar.markdown("## ⚡ **Live Analytics Core**")
    st.sidebar.markdown("---")
    
    # Navigation Radio Buttons with updated terminology
    page = st.sidebar.radio(
        "SYSTEM MODULES",
        [
            "1. Data Ingestion Terminal",
            "2. ETL Pipeline Config",
            "3. Live Model Telemetry",
            "4. BI Command Center",
            "5. Supply Chain Forecasting"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**System Status**: Awaiting Input\n\n"
        "Initialize the dashboard by uploading a dataset in the Data Ingestion Terminal."
    )
    return page


import time

def page_upload_data():
    st.title("📡 Data Ingestion Terminal")
    st.markdown("Initiate real-time data streaming by uploading your CSV/Excel batch file below.")
    
    uploaded_file = st.file_uploader("Initialize Data Feed", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if 'upload_processed' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
                with st.spinner("Ingesting data stream..."):
                    time.sleep(0.8) # Simulate processing for dynamic feel
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state['raw_data'] = df
                    st.session_state['upload_processed'] = True
                    st.session_state['last_uploaded'] = uploaded_file.name
            else:
                df = st.session_state['raw_data']
            
            st.success(f"Stream established: **{uploaded_file.name}**")
            
            # Real-time Metrics Dashboard
            st.markdown("### 📊 Ingestion Telemetry")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{df.shape[0]:,}")
            m2.metric("Total Features", f"{df.shape[1]:,}")
            m3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            m4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.markdown("---")
            col_data, col_types = st.columns([2, 1])
            
            with col_data:
                st.markdown("#### Live Data Feed Preview")
                st.dataframe(df.head(15), use_container_width=True)
                
            with col_types:
                st.markdown("#### Schema Inference")
                dtypes_df = pd.DataFrame(df.dtypes.astype(str), columns=["Data Type"]).reset_index().rename(columns={"index": "Feature"})
                st.dataframe(dtypes_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Stream Interruption: {e}")
    else:
        if st.session_state.get('raw_data') is not None:
            df = st.session_state['raw_data']
            st.info("Active dataset stream detected across session.")
            
            st.markdown("### 📊 Ingestion Telemetry")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Records", f"{df.shape[0]:,}")
            m2.metric("Total Features", f"{df.shape[1]:,}")
            m3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            m4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.markdown("---")
            st.markdown("#### Live Data Feed Preview")
            st.dataframe(df.head(10), use_container_width=True)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import io

def page_preprocessing():
    st.title("⚙️ ETL Pipeline Configuration")
    st.markdown("Configure data transformations, handle anomalies, and prepare the dataset for modeling.")
    
    if st.session_state.get('raw_data') is None:
        st.warning("⚠️ No data stream detected. Please initialize the feed in the Data Ingestion Terminal.")
        return
        
    if st.session_state.get('clean_data') is None:
        df = st.session_state['raw_data'].copy()
    else:
        df = st.session_state['clean_data']

    # Using Tabs for a cleaner pipeline view
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🛠️ Transformation Config", "✂️ Train/Test Split"])

    with tab1:
        st.markdown("#### Current State Snapshot")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Rows", f"{df.shape[0]:,}")
        col_m2.metric("Features", f"{df.shape[1]:,}")
        col_m3.metric("Total Missing Values", f"{df.isnull().sum().sum():,}")
        
        st.markdown("#### Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.markdown("#### Feature Engineering & Cleaning")
        col_clean, col_encode, col_scale = st.columns(3)
        
        with col_clean:
            st.markdown("##### Handle Missing Data")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if not missing_cols:
                st.success("Dataset is clean.")
            else:
                handle_missing = st.selectbox("Imputation Strategy", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
                if st.button("Apply Imputation", key="btn_imp"):
                    with st.spinner("Processing..."):
                        time.sleep(0.5)
                        if handle_missing == "Drop Rows":
                            df = df.dropna()
                        elif handle_missing == "Fill with Mean":
                            df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
                        elif handle_missing == "Fill with Median":
                            df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)
                        elif handle_missing == "Fill with Mode":
                            for col in missing_cols:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                        st.session_state['clean_data'] = df
                        st.success(f"Strategy '{handle_missing}' applied.")
                        time.sleep(1)
                        st.rerun()

        with col_encode:
            st.markdown("##### Categorical Encoding")
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if not cat_cols:
                st.success("No categorical features.")
            else:
                encode_strat = st.selectbox("Encoding Strategy", ["None", "Label Encoding", "One-Hot Encoding"])
                if st.button("Apply Encoding", key="btn_enc"):
                    with st.spinner("Processing..."):
                        time.sleep(0.5)
                        if encode_strat == "Label Encoding":
                            le = LabelEncoder()
                            for col in cat_cols:
                                df[col] = df[col].astype(str)
                                df[col] = le.fit_transform(df[col])
                        elif encode_strat == "One-Hot Encoding":
                            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                        st.session_state['clean_data'] = df
                        st.success(f"Strategy '{encode_strat}' applied.")
                        time.sleep(1)
                        st.rerun()

        with col_scale:
            st.markdown("##### Feature Scaling")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            scale_strat = st.selectbox("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
            if st.button("Apply Scaling", key="btn_scl"):
                with st.spinner("Processing..."):
                    time.sleep(0.5)
                    if scale_strat != "None" and num_cols:
                        if scale_strat == "StandardScaler":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        df[num_cols] = scaler.fit_transform(df[num_cols])
                        st.session_state['clean_data'] = df
                        st.success(f"Strategy '{scale_strat}' applied.")
                        time.sleep(1)
                        st.rerun()

    with tab3:
        st.markdown("#### Training Partition Configuration")
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            target_col = st.selectbox("Target Variable (Y)", df.columns.tolist(), index=len(df.columns)-1)
        with grid_col2:
            test_size = st.slider("Holdout Set Size (%)", 10, 50, 20) / 100.0
            
        if st.button("Initialize Training Split", type="primary"):
            with st.spinner("Partitioning data space..."):
                time.sleep(0.8)
                st.session_state['target_col'] = target_col
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                st.session_state['split_data'] = {
                    'X_train': X_train, 'X_test': X_test,
                    'y_train': y_train, 'y_test': y_test
                }
                
                st.success("✅ Partition established successfully.")
                st.info(f"**Training Elements:** {X_train.shape[0]:,} | **Holdout Elements:** {X_test.shape[0]:,}")

    st.session_state['clean_data'] = df


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def page_models_eval():
    st.title("🤖 Live Model Telemetry")
    st.markdown("Real-time training and performance tracking of predictive models.")
    
    if 'split_data' not in st.session_state:
        st.warning("⚠️ Training data offline. Initialize ETL Pipeline Config first.")
        return
        
    split_data = st.session_state['split_data']
    X_train, X_test = split_data['X_train'], split_data['X_test']
    y_train, y_test = split_data['y_train'], split_data['y_test']
    
    st.info(f"**Data Streams Active** — **Training:** {X_train.shape[0]:,} records | **Holdout:** {X_test.shape[0]:,} records")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }
    
    st.markdown("#### 🎯 Model Selection Matrix")
    selected_models = []
    cols = st.columns(len(models))
    for i, model_name in enumerate(models.keys()):
        with cols[i]:
            if st.checkbox(model_name, value=True):
                selected_models.append(model_name)
                
    if st.button("Initialize Training Sequence", type="primary"):
        if not selected_models:
            st.error("❌ Abort: No models selected.")
            return
            
        with st.spinner("Compiling and training models in parallel..."):
            time.sleep(1.2) # Simulate parallel compute UI
            results = []
            trained_models_dict = {}
            for name in selected_models:
                model = models[name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)
                
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1
                })
                
                trained_models_dict[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'cm': cm
                }
            
            # Save results in session state
            results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
            st.session_state['model_results_df'] = results_df
            st.session_state['trained_models'] = trained_models_dict
            
            # Find the best model
            best_model_name = results_df.iloc[0]["Model"]
            st.session_state['best_model_name'] = best_model_name
            st.session_state['best_model'] = trained_models_dict[best_model_name]['model']
            
        st.success("✅ Sequence Complete: Models compiled and evaluated.")
            
    # Display results if available
    if 'model_results_df' in st.session_state:
        results_df = st.session_state['model_results_df']
        trained_models = st.session_state['trained_models']
        best_model_name = st.session_state['best_model_name']
        
        st.markdown("---")
        st.markdown("#### 📊 Performance Telemetry")
        
        # Display as a styled dataframe
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='#31333F'), use_container_width=True)
        st.success(f"🏆 System Recommendation: **{best_model_name}** achieved peak accuracy.")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            # Accuracy Bar Chart
            fig = px.bar(results_df, x='Model', y='Accuracy', color='Model', title="Global Accuracy Leaderboard", text_auto='.3f', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_c2:
            # Multi-metric Radar Chart for Model Comparison
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            fig_radar = go.Figure()
            for i, row in results_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score']],
                    theta=categories,
                    fill='toself',
                    name=row['Model']
                ))
            fig_radar.update_layout(
                template="plotly_dark", 
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.2)"),
                    bgcolor="rgba(0,0,0,0)"
                ), 
                paper_bgcolor="rgba(0,0,0,0)",
                title="Multivariate Capabilities (Radar)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 🔍 Deep Diagnostics")
        
        viz_model = st.selectbox("Select Model for Deep Diagnosis", results_df['Model'].tolist())
        model_data = trained_models[viz_model]
        
        t1, t2, t3 = st.tabs(["Confusion Matrix & Splits", "ROC & PR Curves", "Feature Attribution"])
        
        with t1:
            col1, col2 = st.columns(2)
            with col1:
                fig_cm = px.imshow(model_data['cm'], text_auto=True, color_continuous_scale='Blues',
                                  labels=dict(x="Predicted Class", y="True Class", color="Count"), title="Confusion Matrix")
                fig_cm.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with col2:
                correct_preds = np.trace(model_data['cm'])
                wrong_preds = np.sum(model_data['cm']) - correct_preds
                fig_pie = px.pie(values=[correct_preds, wrong_preds], names=['Success', 'Failure'], 
                                 title="Prediction Outcome Distribution", color_discrete_sequence=['#00CC96', '#EF553B'])
                fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with t2:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            if len(np.unique(y_test)) == 2 and model_data['y_proba'] is not None:
                col_r1, col_r2 = st.columns(2)
                y_test_bin = (y_test == np.unique(y_test)[1]).astype(int)
                
                with col_r1:
                    fpr, tpr, thresholds = roc_curve(y_test_bin, model_data['y_proba'])
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:0.2f})', line=dict(color='#00CC96')))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random'))
                    fig_roc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="ROC Curve", xaxis_title='FPR', yaxis_title='TPR')
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                with col_r2:
                    precision, recall, _ = precision_recall_curve(y_test_bin, model_data['y_proba'])
                    pr_auc = average_precision_score(y_test_bin, model_data['y_proba'])
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:0.2f})', line=dict(color='#AB63FA')))
                    fig_pr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title="Precision-Recall Curve", xaxis_title='Recall', yaxis_title='Precision')
                    st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("📉 Probability Curves (ROC/PR) are reserved for Binary Classification models with probability outputs.")
                
        with t3:
            if viz_model in ["Decision Tree", "Random Forest"]:
                importances = model_data['model'].feature_importances_
                feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                fig_feat = px.bar(feat_imp.tail(10), x='Importance', y='Feature', orientation='h', title=f"Dominant Attributes ({viz_model})", color='Importance', color_continuous_scale='Magma')
                fig_feat.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("📉 Feature Attribution is inherently extracted from Tree-based methodologies (Decision Tree, Random Forest).")

import joblib

def page_bi_insights():
    st.title("📈 BI Command Center")
    st.markdown("Translate raw predictions into actionable business strategy and export AI agents.")
    
    if 'best_model_name' not in st.session_state or st.session_state['best_model_name'] is None:
        st.warning("⚠️ Telemetry unavailable. Complete Model Telemetry phase first.")
        return
        
    best_model_name = st.session_state['best_model_name']
    best_model = st.session_state['best_model']
    results_df = st.session_state['model_results_df']
    split_data = st.session_state['split_data']
    target_col = st.session_state['target_col']
    X_train = split_data['X_train']
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("#### 🧠 Prime Diagnostic")
        f1 = results_df.iloc[0]['F1 Score']
        if f1 > 0.85:
            st.success(f"**Optimum Confidence:** The selected {best_model_name} operates with a robust F1-core of {f1:.2f}. Predictions are certified for automated business deployment.")
        elif f1 > 0.70:
            st.warning(f"**Moderate Confidence:** The {best_model_name} indicates an F1-score of {f1:.2f}. Human-in-the-loop review implies best outcomes.")
        else:
            st.error(f"**Low Confidence:** System F1-score is {f1:.2f}. High risk anomaly. Augment data prior to operational execution.")
    with col_b:
        st.markdown("#### 💾 Agent Extraction")
        buffer = io.BytesIO()
        joblib.dump(best_model, buffer)
        buffer.seek(0)
        st.download_button(
            label=f"⬇️ Download {best_model_name} (.pkl)",
            data=buffer,
            file_name=f"{best_model_name.replace(' ', '_')}_v1.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 💡 Causal Discoveries")
    
    if best_model_name in ["Decision Tree", "Random Forest"]:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = X_train.columns[indices][:3].tolist()
        st.info(f"The primary vectors dictating **{target_col}** anomalies are **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**. Optimize operational capital here.")
    elif best_model_name in ["Logistic Regression", "Support Vector Machine"] and hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_[0])
        indices = np.argsort(coefs)[::-1]
        if len(indices) >= 3:
            top_features = X_train.columns[indices][:3].tolist()
            st.info(f"System variance is fundamentally chained to **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**.")
    else:
        st.info("Model relies on complex manifold topologies. Variables are entangled in non-linear multi-dimensional hyperspaces.")

    st.markdown("---")
    st.markdown("#### 📥 Batch Inference Sandbox")
    st.write("Stream unlabelled zero-day data through the selected AI pipeline.")
    
    test_file = st.file_uploader("Upload Target Data (CSV/Excel)", type=["csv", "xlsx", "xls"], key="test_upload")
    
    if test_file is not None:
        try:
            with st.spinner("Processing zero-day stream..."):
                time.sleep(1)
                if test_file.name.endswith('.csv'):
                    test_df = pd.read_csv(test_file)
                else:
                    test_df = pd.read_excel(test_file)
                
                st.dataframe(test_df.head(), use_container_width=True)
                
                missing_cols = set(X_train.columns) - set(test_df.columns)
                if missing_cols:
                    st.error(f"Schema mismatch. Missing attributes: {missing_cols}")
                else:
                    X_new = test_df[X_train.columns]
                    if X_new.isnull().sum().sum() > 0:
                        st.warning("Handling NULL anomalies using zero-fill protocol.")
                        X_new = X_new.fillna(0)
                    
                    if st.button("Engage Predictor", type="primary"):
                        time.sleep(0.5)
                        predictions = best_model.predict(X_new)
                        test_df['AutoML_Prediction_' + target_col] = predictions
                        
                        st.success("✅ Output rendered.")
                        st.dataframe(test_df.head(10), use_container_width=True)
                        
                        csv = test_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Export Inference Data",
                            data=csv,
                            file_name="automl_inferences.csv",
                            mime="text/csv",
                        )
        except Exception as e:
            st.error(f"Integrity Error: {e}")

def page_supply_chain():
    st.title("📦 Supply Chain Forecasting")
    st.markdown("Stochastic demand projections and inventory risk algorithms.")
    
    if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
        st.warning("⚠️ Warehouse variables missing. Execute ETL Config phase.")
        return
        
    df = st.session_state['clean_data']
    st.markdown("#### Inventory Head")
    st.dataframe(df.head(), use_container_width=True)
    
    st.markdown("---")
    
    t_fc, t_risk = st.tabs(["📉 Temporal Demand Vectors", "⚠️ Volatility Analysis"])
    
    with t_fc:
        st.markdown("##### Temporal Pattern Matching (SMA/EMA)")
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox("Temporal Index (Optional)", ["None"] + df.columns.tolist())
        with col2:
            demand_col = st.selectbox("Consumption Vector", df.columns.tolist(), index=len(df.columns)-1)
        with col3:
            window = st.slider("Lookback Window", min_value=2, max_value=30, value=7)
            
        if st.button("Compute Forecast Matrix", type="primary"):
            with st.spinner("Calculating temporal drifts..."):
                time.sleep(0.5)
                temp_df = df.copy()
                if date_col != "None" and date_col in temp_df.columns:
                    try:
                        temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                        temp_df = temp_df.sort_values(by=date_col)
                    except Exception:
                        st.warning(f"Temporal parsing compromised for {date_col}.")
                        
                temp_df['SMA (Moving Avg)'] = temp_df[demand_col].rolling(window=window).mean()
                temp_df['EMA (Exp Smoothing)'] = temp_df[demand_col].ewm(span=window, adjust=False).mean()
                
                fig = go.Figure()
                x_axis = temp_df[date_col] if date_col != "None" else temp_df.index
                
                fig.add_trace(go.Scatter(x=x_axis, y=temp_df[demand_col], mode='lines', name='Actual Flow', opacity=0.4, line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=x_axis, y=temp_df['SMA (Moving Avg)'], mode='lines', name=f'{window}-SMA', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=x_axis, y=temp_df['EMA (Exp Smoothing)'], mode='lines', name=f'{window}-EMA', line=dict(color='#AB63FA')))
                
                fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", title="Macro Demand Oscillation", xaxis_title="Timeline", yaxis_title="Volume")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("System Note: Exponential smoothing maps accurately to recency bias in supply chain consumption.")

    with t_risk:
        st.markdown("##### Inventory Risk Heatmap")
        cat_col = st.selectbox("SKU/Product Descriptor", ["None"] + df.select_dtypes(exclude=[np.number]).columns.tolist())
        
        if cat_col != "None":
            if st.button("Synthesize Risk Data"):
                with st.spinner("Processing volatility limits..."):
                    time.sleep(0.5)
                    risk_df = df.groupby(cat_col)[demand_col].agg(['sum', 'std', 'count']).dropna()
                    risk_df = risk_df.rename(columns={'sum': 'Aggregate Volume', 'std': 'Standard Deviation (Risk)'})
                    risk_df = risk_df.reset_index()
                    
                    fig2 = px.scatter(risk_df, x='Aggregate Volume', y='Standard Deviation (Risk)', color='Standard Deviation (Risk)', 
                                      size='count', hover_name=cat_col, color_continuous_scale="Inferno",
                                      title="Risk Distribution Matrix")
                    fig2.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)
                    st.error("Protocol: SKUs trending Top-Right represent hyper-volatile core requirements. Increment safety stock margins.")

def main():
    # Render Sidebar and get current page
    current_page = render_sidebar()

    # Route to the appropriate page function
    if current_page == "1. Data Ingestion Terminal":
        page_upload_data()
    elif current_page == "2. ETL Pipeline Config":
        page_preprocessing()
    elif current_page == "3. Live Model Telemetry":
        page_models_eval()
    elif current_page == "4. BI Command Center":
        page_bi_insights()
    elif current_page == "5. Supply Chain Forecasting":
        page_supply_chain()

if __name__ == "__main__":
    main()
