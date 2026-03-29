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

# Custom CSS for clean UI/UX
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2c3e50;
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
    st.sidebar.title("🧠 AutoML Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation Radio Buttons
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Upload Data",
            "2. Preprocessing",
            "3. Models & Evaluation",
            "4. BI Insights",
            "5. Supply Chain Analytics"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Upload a dataset to get started. Navigate through the steps to preprocess, "
        "train models, and view business insights."
    )
    return page


def page_upload_data():
    st.title("📂 Step 1: Upload Dataset")
    st.markdown("Upload your CSV or Excel dataset to begin the analysis.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the file based on its extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.session_state['raw_data'] = df
            st.success("Dataset uploaded successfully!")
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            # Display dataset info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Total Records:**", df.shape[0])
                st.write("**Total Features:**", df.shape[1])
            with col2:
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())
            
            st.subheader("Data Types")
            st.write(df.dtypes.astype(str))
            
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        # If there's already data in session state, show it
        if st.session_state['raw_data'] is not None:
            df = st.session_state['raw_data']
            st.info("Using previously uploaded dataset.")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Dataset Shape:**", df.shape)
            with col2:
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import io

def page_preprocessing():
    st.title("⚙️ Step 2: Data Preprocessing")
    st.markdown("Handle missing values, encode features, and split the data for training.")
    
    if st.session_state['raw_data'] is None:
        st.warning("Please upload a dataset in Step 1 first.")
        return
        
    # We work on a copy to avoid mutating the uploaded file unpredictably
    if st.session_state['clean_data'] is None:
        df = st.session_state['raw_data'].copy()
    else:
        df = st.session_state['clean_data']
    
    st.subheader("Current Dataset Preview")
    st.dataframe(df.head(5))

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            st.success("No missing values found in the dataset!")
        else:
            st.write(f"Columns with missing values: {', '.join(missing_cols)}")
            handle_missing = st.selectbox("Select strategy", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"])
            
            if st.button("Apply Missing Value Strategy"):
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
                st.success(f"Applied '{handle_missing}' to missing values.")
                st.rerun()

    with col2:
        st.subheader("2. Encode Categorical Variables")
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not cat_cols:
            st.success("No categorical columns found.")
        else:
            st.write(f"Categorical columns: {', '.join(cat_cols)}")
            encode_strat = st.selectbox("Select encoding", ["None", "Label Encoding", "One-Hot Encoding"])
            
            if st.button("Apply Encoding"):
                if encode_strat == "Label Encoding":
                    le = LabelEncoder()
                    for col in cat_cols:
                        df[col] = df[col].astype(str)
                        df[col] = le.fit_transform(df[col])
                elif encode_strat == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                
                st.session_state['clean_data'] = df
                st.success(f"Applied '{encode_strat}'.")
                st.rerun()

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("3. Feature Scaling")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_strat = st.selectbox("Select scaling method", ["None", "StandardScaler", "MinMaxScaler"])
        
        if st.button("Apply Scaling"):
            if scale_strat != "None" and num_cols:
                if scale_strat == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                df[num_cols] = scaler.fit_transform(df[num_cols])
                st.session_state['clean_data'] = df
                st.success(f"Applied '{scale_strat}'.")
                st.rerun()

    with col4:
        st.subheader("4. Train-Test Split")
        st.write("Select target variable and test size.")
        
        target_col = st.selectbox("Target Variable (Y)", df.columns.tolist(), index=len(df.columns)-1)
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100.0
        
        if st.button("Generate Train/Test Split"):
            st.session_state['target_col'] = target_col
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.session_state['split_data'] = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }
            
            st.success(f"Data split successfully! Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows.")

    # Always ensure the clean data is saved if we made it here
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
    st.title("🤖 Step 3: Models & Evaluation")
    st.markdown("Train machine learning models and compare their performance.")
    
    if 'split_data' not in st.session_state:
        st.warning("Please preprocess and split the data in Step 2 first.")
        return
        
    split_data = st.session_state['split_data']
    X_train, X_test = split_data['X_train'], split_data['X_test']
    y_train, y_test = split_data['y_train'], split_data['y_test']
    
    st.markdown(f"**Training Set:** {X_train.shape[0]} samples | **Test Set:** {X_test.shape[0]} samples")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }
    
    st.subheader("Select Models to Train")
    selected_models = []
    cols = st.columns(len(models))
    for i, model_name in enumerate(models.keys()):
        with cols[i]:
            if st.checkbox(model_name, value=True):
                selected_models.append(model_name)
                
    if st.button("Train Selected Models & Evaluate"):
        if not selected_models:
            st.error("Please select at least one model to train.")
            return
            
        with st.spinner("Training models..."):
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
            
        st.success("Training Complete!")
            
    # Display results if available
    if 'model_results_df' in st.session_state:
        results_df = st.session_state['model_results_df']
        trained_models = st.session_state['trained_models']
        best_model_name = st.session_state['best_model_name']
        
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison")
        
        # Display as a styled dataframe
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='lightgreen'))
        st.info(f"🏆 Best Model based on Accuracy: **{best_model_name}**")
        
        # Accuracy Bar Chart
        fig = px.bar(results_df, x='Model', y='Accuracy', color='Model', title="Model Accuracy Comparison", text_auto='.3f')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric Radar Chart for Model Comparison
        st.markdown('**Multi-Metric Comparison (Radar Chart)**')
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        fig_radar = go.Figure()
        for i, row in results_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score']],
                theta=categories,
                fill='toself',
                name=row['Model']
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Performance Radar Chart")
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Detailed Model Visualizations")
        
        viz_model = st.selectbox("Select model for details", results_df['Model'].tolist())
        model_data = trained_models[viz_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            st.write("**Confusion Matrix Heatmap**")
            fig_cm = px.imshow(model_data['cm'], text_auto=True, color_continuous_scale='Blues',
                              labels=dict(x="Predicted Label", y="True Label", color="Count"))
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Prediction Breakdown Pie Chart
            st.write("**Prediction Success Breakdown (Pie Chart)**")
            correct_preds = np.trace(model_data['cm'])
            wrong_preds = np.sum(model_data['cm']) - correct_preds
            fig_pie = px.pie(values=[correct_preds, wrong_preds], names=['Correct Predictions', 'Incorrect Predictions'], 
                             title="Overall Correct vs Incorrect Ratio", color_discrete_sequence=['#4CAF50', '#EF5350'])
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # ROC Curve and PR Curve (if binary classification and probabilities available)
            if len(np.unique(y_test)) == 2 and model_data['y_proba'] is not None:
                st.write("**ROC Curve & Precision-Recall Curve**")
                y_test_bin = (y_test == np.unique(y_test)[1]).astype(int)
                
                # ROC
                fpr, tpr, thresholds = roc_curve(y_test_bin, model_data['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:0.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random', opacity=0.5))
                fig_roc.update_layout(title="Receiver Operating Characteristic (ROC)", xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # PR Curve
                precision, recall, _ = precision_recall_curve(y_test_bin, model_data['y_proba'])
                pr_auc = average_precision_score(y_test_bin, model_data['y_proba'])
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR curve (area = {pr_auc:0.2f})'))
                fig_pr.update_layout(title="Precision-Recall Curve (Line Chart)", xaxis_title='Recall', yaxis_title='Precision')
                st.plotly_chart(fig_pr, use_container_width=True)
                
            else:
                st.write("**ROC & PR Curves**")
                st.info("Line curves (ROC/PR) are available only for binary classification models with probability output.")
                
        # Feature Importance for Tree models
        if viz_model in ["Decision Tree", "Random Forest"]:
            st.write("**Feature Importance (Horizontal Bar Chart)**")
            importances = model_data['model'].feature_importances_
            feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
            fig_feat = px.bar(feat_imp.tail(10), x='Importance', y='Feature', orientation='h', title=f"Top 10 Feature Importances ({viz_model})")
            st.plotly_chart(fig_feat, use_container_width=True)


import joblib

def page_bi_insights():
    st.title("📈 Step 4: Business Intelligence (BI) Insights")
    st.markdown("Actionable insights derived from the best performing model.")
    
    if 'best_model_name' not in st.session_state or st.session_state['best_model_name'] is None:
        st.warning("Please train and evaluate models in Step 3 first.")
        return
        
    best_model_name = st.session_state['best_model_name']
    best_model = st.session_state['best_model']
    results_df = st.session_state['model_results_df']
    split_data = st.session_state['split_data']
    target_col = st.session_state['target_col']
    X_train = split_data['X_train']
    
    st.success(f"🏆 Our analysis selected **{best_model_name}** as the best model with an accuracy of **{results_df.iloc[0]['Accuracy']:.2%}**.")
    
    st.subheader("💡 Key Business Insights")
    
    # 1. Influence of features
    if best_model_name in ["Decision Tree", "Random Forest"]:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = X_train.columns[indices][:3].tolist()
        
        st.markdown(f"**1. What drives '{target_col}'?**")
        st.info(f"The most important factors influencing **{target_col}** are **{top_features[0]}**, followed by **{top_features[1]}** and **{top_features[2]}**. Focusing business strategies on these areas will yield the highest impact.")
    elif best_model_name in ["Logistic Regression", "Support Vector Machine"] and hasattr(best_model, "coef_"):
        # For linear models
        coefs = np.abs(best_model.coef_[0])
        indices = np.argsort(coefs)[::-1]
        top_features = X_train.columns[indices][:3].tolist()
        
        st.markdown(f"**1. What drives '{target_col}'?**")
        st.info(f"The most impactful features determining the outcome are **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**. Any changes in these variables will strongly affect the result.")
    else:
        st.markdown(f"**1. General Trend for '{target_col}'**")
        st.info("The selected model uses complex distance or non-linear patterns. While exact feature importance is harder to extract, the model reliably predicts outcomes based on the holistic combination of all variables.")

    # 2. Performance Context
    st.markdown("**2. Reliability of Predictions**")
    f1 = results_df.iloc[0]['F1 Score']
    if f1 > 0.85:
        st.success(f"The model's F1-score is very high ({f1:.2f}). You can highly trust these automated predictions in real-world scenarios.")
    elif f1 > 0.70:
        st.warning(f"The model has a moderate F1-score ({f1:.2f}). It is useful for general trends but should be combined with human judgment for critical decisions.")
    else:
        st.error(f"The model's performance is relatively low ({f1:.2f}). Consider collecting more data, adding new features, or doing deeper data cleaning before relying on this for production decisions.")
        
    st.markdown("---")
    
    st.subheader("💾 Export Best Model")
    st.write("Download the trained best model to deploy it in your own applications.")
    
    # Save the model to a bytes buffer
    buffer = io.BytesIO()
    joblib.dump(best_model, buffer)
    buffer.seek(0)
    
    st.download_button(
        label="⬇️ Download Trained Model (.pkl)",
        data=buffer,
        file_name=f"{best_model_name.replace(' ', '_')}_model.pkl",
        mime="application/octet-stream"
    )

    st.markdown("---")
    st.subheader("🔮 Predict on New Data")
    st.write("Upload a new dataset (without the target column) to generate predictions.")
    
    test_file = st.file_uploader("Upload New Data (CSV/Excel)", type=["csv", "xlsx", "xls"], key="test_upload")
    
    if test_file is not None:
        try:
            if test_file.name.endswith('.csv'):
                test_df = pd.read_csv(test_file)
            else:
                test_df = pd.read_excel(test_file)
                
            st.write("New Data Preview:")
            st.dataframe(test_df.head())
            
            # Check if columns match
            missing_cols = set(X_train.columns) - set(test_df.columns)
            if missing_cols:
                st.error(f"Missing columns in uploaded data: {missing_cols}")
            else:
                # Subset to ensure same order
                X_new = test_df[X_train.columns]
                
                # Check for missing values
                if X_new.isnull().sum().sum() > 0:
                    st.warning("New data contains missing values. Filling with 0. For best results, preprocess new data similarly to training data.")
                    X_new = X_new.fillna(0)
                
                if st.button("Generate Predictions"):
                    predictions = best_model.predict(X_new)
                    test_df['Predicted_' + target_col] = predictions
                    
                    st.success("Predictions generated!")
                    st.dataframe(test_df.head(10))
                    
                    # Convert to CSV for download
                    csv = test_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing new data: {e}")


def page_supply_chain():
    st.title("📦 Step 5: Supply Chain Analytics")
    st.markdown("Demand Forecasting & Inventory Risk features tailored for Supply Chain Management.")
    
    if 'clean_data' not in st.session_state or st.session_state['clean_data'] is None:
        st.warning("Please upload and preprocess data first to use Supply Chain tools.")
        return
        
    df = st.session_state['clean_data']
    st.dataframe(df.head())
    
    st.markdown("---")
    st.subheader("📈 Demand Forecasting")
    st.write("Forecast future demand using Moving Averages and Exponential Smoothing.")
    
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Select Date/Time Column (Optional but recommended)", ["None"] + df.columns.tolist())
    with col2:
        demand_col = st.selectbox("Select Demand/Sales Column", df.columns.tolist(), index=len(df.columns)-1)
        
    window = st.slider("Rolling Window Size", min_value=2, max_value=30, value=7)
    
    if st.button("Generate Forecast View"):
        temp_df = df.copy()
        
        # Sort by date if available
        if date_col != "None" and date_col in temp_df.columns:
            try:
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                temp_df = temp_df.sort_values(by=date_col)
            except Exception:
                st.warning(f"Could not convert {date_col} to datetime. Plotting without datetime scaling.")
                
        # Calculate Moving Average & Exponential Smoothing
        temp_df['SMA (Moving Avg)'] = temp_df[demand_col].rolling(window=window).mean()
        temp_df['EMA (Exp Smoothing)'] = temp_df[demand_col].ewm(span=window, adjust=False).mean()
        
        fig = go.Figure()
        
        x_axis = temp_df[date_col] if date_col != "None" else temp_df.index
        
        fig.add_trace(go.Scatter(x=x_axis, y=temp_df[demand_col], mode='lines', name='Actual Demand', opacity=0.5))
        fig.add_trace(go.Scatter(x=x_axis, y=temp_df['SMA (Moving Avg)'], mode='lines', name=f'{window}-Period SMA'))
        fig.add_trace(go.Scatter(x=x_axis, y=temp_df['EMA (Exp Smoothing)'], mode='lines', name=f'{window}-Period EMA'))
        
        fig.update_layout(title="Demand Forecasting Trends", xaxis_title="Time/Index", yaxis_title="Demand")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** Moving Averages smooth out short-term fluctuations and highlight longer-term supply chain trends.")

    st.markdown("---")
    st.subheader("⚠️ Inventory Volatility & Risk Analysis")
    st.write("Analyze the volatility (standard deviation) of features compared to their volume to identify risky inventory.")
    
    cat_col = st.selectbox("Select Category/Product Identifier", ["None"] + df.select_dtypes(exclude=[np.number]).columns.tolist())
    
    if cat_col != "None" and st.button("Analyze Inventory Risk"):
        # Group by category
        risk_df = df.groupby(cat_col)[demand_col].agg(['sum', 'std', 'count']).dropna()
        risk_df = risk_df.rename(columns={'sum': 'Total Demand', 'std': 'Demand Volatility'})
        risk_df = risk_df.reset_index()
        
        fig2 = px.scatter(risk_df, x='Total Demand', y='Demand Volatility', color='Demand Volatility', 
                          size='count', hover_name=cat_col, 
                          title="Inventory Risk Matrix (Volatility vs Total Demand)")
        st.plotly_chart(fig2, use_container_width=True)
        st.info("💡 **Insight:** Items in the top-right quadrant have high demand but high volatility, making them the most challenging for inventory planning. Consider maintaining higher safety safety stock for these items.")

def main():
    # Render Sidebar and get current page
    current_page = render_sidebar()

    # Route to the appropriate page function
    if current_page == "1. Upload Data":
        page_upload_data()
    elif current_page == "2. Preprocessing":
        page_preprocessing()
    elif current_page == "3. Models & Evaluation":
        page_models_eval()
    elif current_page == "4. BI Insights":
        page_bi_insights()
    elif current_page == "5. Supply Chain Analytics":
        page_supply_chain()

if __name__ == "__main__":
    main()
