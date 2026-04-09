import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Medical Expense ML Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>🏥 Personal Medical Expense ML Pipeline</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'problem_type' not in st.session_state:
        st.session_state.problem_type = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []

    # Horizontal step indicators
    steps = ["Problem Type", "Data Input", "EDA", "Data Engineering", "Feature Selection",
             "Data Split", "Model Selection", "Training & Validation", "Performance Metrics", "Hyperparameter Tuning"]

    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 <= st.session_state.step:
                st.markdown(f"<div style='background-color:#1f77b4;color:white;padding:10px;border-radius:5px;text-align:center;font-weight:bold;'>{i+1}. {step_name}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:#e9ecef;color:#6c757d;padding:10px;border-radius:5px;text-align:center;'>{i+1}. {step_name}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Step 1: Problem Type Selection
    if st.session_state.step == 1:
        st.markdown("<h2 class='step-header'>Step 1: Select Problem Type</h2>", unsafe_allow_html=True)
        st.write("Choose whether you want to solve a Classification or Regression problem for your medical expense data.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Classification", key="classification"):
                st.session_state.problem_type = "Classification"
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("📈 Regression", key="regression"):
                st.session_state.problem_type = "Regression"
                st.session_state.step = 2
                st.rerun()

        st.info("💡 **Classification**: Predict categories (e.g., high/low medical expense, disease presence)\n\n**Regression**: Predict continuous values (e.g., exact medical expense amount)")

    # Step 2: Data Input
    elif st.session_state.step == 2:
        st.markdown("<h2 class='step-header'>Step 2: Data Input</h2>", unsafe_allow_html=True)
        st.write(f"**Problem Type:** {st.session_state.problem_type}")

        uploaded_file = st.file_uploader("Upload your medical expense dataset (CSV file)", type=['csv'])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")

                st.subheader("Data Preview")
                st.dataframe(data.head(10))

                # Target selection
                st.subheader("Select Target Feature")
                target_column = st.selectbox("Choose the column you want to predict:", data.columns.tolist())
                st.session_state.target_column = target_column

                # Feature selection
                st.subheader("Select Features")
                feature_cols = [col for col in data.columns if col != target_column]
                selected_features = st.multiselect("Select features for your model:", feature_cols, default=feature_cols)
                st.session_state.selected_features = selected_features

                # PCA Visualization
                st.subheader("PCA Visualization of Data Shape")
                numeric_data = data[selected_features].select_dtypes(include=[np.number])
                if numeric_data.shape[1] >= 2:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data.fillna(numeric_data.mean()))
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)

                    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1],
                                   title="2D PCA Projection of Your Data",
                                   labels={'x': 'PC1', 'y': 'PC2'},
                                   color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig, use_container_width=True)

                    # Explained variance
                    st.write(f"**Explained Variance:** PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")

                st.write(f"**Selected Data Shape:** {len(selected_features)} features × {len(data)} samples")

                if st.button("Proceed to EDA →", key="proceed_eda"):
                    st.session_state.step = 3
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        if st.button("← Go Back", key="back_step2"):
            st.session_state.step = 1
            st.rerun()

    # Step 3: EDA
    elif st.session_state.step == 3:
        st.markdown("<h2 class='step-header'>Step 3: Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)

        if st.session_state.data is not None:
            data = st.session_state.data
            selected_features = st.session_state.selected_features
            target = st.session_state.target_column

            # Basic Statistics
            st.subheader("📊 Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
            with col4:
                st.metric("Categorical Features", len(data.select_dtypes(include=['object']).columns))

            # Missing Values
            st.subheader("❓ Missing Values Analysis")
            missing = data.isnull().sum()
            missing_df = pd.DataFrame({'Column': missing.index, 'Missing Values': missing.values, 'Percentage': (missing.values / len(data)) * 100})
            missing_df = missing_df[missing_df['Missing Values'] > 0]

            if not missing_df.empty:
                fig = px.bar(missing_df, x='Column', y='Percentage', title="Missing Values by Column",
                            color_discrete_sequence=['#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            # Data Types Distribution
            st.subheader("📝 Data Types Distribution")
            dtype_counts = data.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            fig = px.pie(dtype_counts, values='Count', names='Data Type', title="Data Types Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Numeric Features Distribution
            numeric_cols = data[selected_features].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Numeric Features Distribution")
                selected_num = st.selectbox("Select feature to visualize:", numeric_cols, key="num_feature")

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(data, x=selected_num, title=f"Distribution of {selected_num}",
                                     marginal="box", color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(data, y=selected_num, title=f"Box Plot of {selected_num}",
                                color_discrete_sequence=['#2ecc71'])
                    st.plotly_chart(fig, use_container_width=True)

            # Correlation Matrix
            if len(numeric_cols) > 1:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = data[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Feature Correlation Heatmap",
                              color_continuous_scale='RdBu', aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

            # Target Analysis
            st.subheader(f"🎯 Target Variable Analysis: {target}")
            if st.session_state.problem_type == "Classification":
                fig = px.bar(data[target].value_counts().reset_index(), x='index', y=target,
                           title=f"Class Distribution of {target}", color_discrete_sequence=['#9b59b6'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(data, x=target, title=f"Distribution of {target}",
                                 marginal="box", color_discrete_sequence=['#e67e22'])
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Proceed to Data Engineering →", key="proceed_de"):
                st.session_state.step = 4
                st.rerun()

        if st.button("← Go Back", key="back_step3"):
            st.session_state.step = 2
            st.rerun()

    # Step 4: Data Engineering
    elif st.session_state.step == 4:
        st.markdown("<h2 class='step-header'>Step 4: Data Engineering & Cleaning</h2>", unsafe_allow_html=True)

        if st.session_state.data is not None:
            data = st.session_state.data.copy()

            # Missing Value Handling
            st.subheader("🔧 Missing Value Handling")
            missing_cols = data.columns[data.isnull().any()].tolist()

            if missing_cols:
                st.write("Columns with missing values:", missing_cols)

                handle_method = st.selectbox("Choose method to handle missing values:",
                                           ["Mean", "Median", "Mode", "Drop Rows"])

                if st.button("Apply Missing Value Treatment"):
                    for col in missing_cols:
                        if data[col].dtype in ['int64', 'float64']:
                            if handle_method == "Mean":
                                data[col].fillna(data[col].mean(), inplace=True)
                            elif handle_method == "Median":
                                data[col].fillna(data[col].median(), inplace=True)
                            elif handle_method == "Mode":
                                data[col].fillna(data[col].mode()[0], inplace=True)
                        else:
                            data[col].fillna(data[col].mode()[0], inplace=True)

                    if handle_method == "Drop Rows":
                        data.dropna(inplace=True)

                    st.session_state.processed_data = data
                    st.success(f"✅ Missing values handled using {handle_method}!")
                    st.rerun()
            else:
                st.success("✅ No missing values to handle!")
                st.session_state.processed_data = data

            # Outlier Detection
            st.subheader("🔍 Outlier Detection")
            outlier_method = st.selectbox("Choose outlier detection method:",
                                        ["IQR", "Isolation Forest", "DBSCAN", "Local Outlier Factor"])

            numeric_data = data[st.session_state.selected_features].select_dtypes(include=[np.number])

            if st.button("Detect Outliers"):
                outliers = None

                if outlier_method == "IQR":
                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
                    outliers = outlier_mask

                elif outlier_method == "Isolation Forest":
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers_pred = iso_forest.fit_predict(numeric_data.fillna(numeric_data.mean()))
                    outliers = outliers_pred == -1

                elif outlier_method == "DBSCAN":
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data.fillna(numeric_data.mean()))
                    dbscan = DBSCAN(eps=0.5, min_samples=5)
                    clusters = dbscan.fit_predict(scaled_data)
                    outliers = clusters == -1

                elif outlier_method == "Local Outlier Factor":
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(numeric_data.fillna(numeric_data.mean()))
                    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                    outliers_pred = lof.fit_predict(scaled_data)
                    outliers = outliers_pred == -1

                if outliers is not None:
                    outlier_indices = data.index[outliers].tolist()
                    st.warning(f"⚠️ Found {len(outlier_indices)} outliers using {outlier_method}")

                    if len(outlier_indices) > 0:
                        outlier_df = data.loc[outlier_indices]
                        st.dataframe(outlier_df)

                        remove_outliers = st.checkbox("Remove detected outliers?", key="remove_outliers")
                        if remove_outliers and st.button("Confirm Removal"):
                            data = data.drop(outlier_indices)
                            st.session_state.processed_data = data
                            st.success(f"✅ Removed {len(outlier_indices)} outliers!")
                            st.rerun()

            # Display current processed data
            if st.session_state.processed_data is not None:
                st.subheader("📋 Processed Data Preview")
                st.write(f"Shape: {st.session_state.processed_data.shape}")
                st.dataframe(st.session_state.processed_data.head())

            if st.button("Proceed to Feature Selection →", key="proceed_fs"):
                st.session_state.step = 5
                st.rerun()

        if st.button("← Go Back", key="back_step4"):
            st.session_state.step = 3
            st.rerun()

    # Step 5: Feature Selection
    elif st.session_state.step == 5:
        st.markdown("<h2 class='step-header'>Step 5: Feature Selection</h2>", unsafe_allow_html=True)

        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            selected_features = st.session_state.selected_features
            target = st.session_state.target_column

            st.subheader("🎯 Feature Selection Methods")

            # Variance Threshold
            st.write("#### Variance Threshold")
            variance_threshold = st.slider("Variance Threshold:", 0.0, 1.0, 0.1, 0.01)

            if st.button("Apply Variance Threshold"):
                numeric_data = data[selected_features].select_dtypes(include=[np.number])
                selector = VarianceThreshold(threshold=variance_threshold)
                selector.fit(numeric_data.fillna(numeric_data.mean()))
                selected_var = numeric_data.columns[selector.get_support()].tolist()
                st.write(f"Features passing variance threshold: {selected_var}")

            # Correlation with Target
            st.write("#### Correlation Analysis")
            if st.button("Show Feature Correlations"):
                numeric_data = data[selected_features].select_dtypes(include=[np.number])
                if st.session_state.problem_type == "Regression":
                    correlations = numeric_data.corrwith(data[target]).abs().sort_values(ascending=False)
                else:
                    correlations = numeric_data.apply(lambda x: abs(x.corr(data[target].astype('category').cat.codes))).sort_values(ascending=False)

                fig = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                           title="Feature Correlation with Target", color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)

            # Information Gain
            st.write("#### Information Gain / Mutual Information")
            if st.button("Calculate Information Gain"):
                numeric_data = data[selected_features].select_dtypes(include=[np.number])
                X = numeric_data.fillna(numeric_data.mean())
                y = data[target]

                if st.session_state.problem_type == "Classification":
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    ig_scores = mutual_info_classif(X, y)
                else:
                    ig_scores = mutual_info_regression(X, y)

                ig_df = pd.DataFrame({'Feature': numeric_data.columns, 'Information Gain': ig_scores})
                ig_df = ig_df.sort_values('Information Gain', ascending=True)

                fig = px.bar(ig_df, x='Information Gain', y='Feature', orientation='h',
                           title="Information Gain by Feature", color_discrete_sequence=['#2ecc71'])
                st.plotly_chart(fig, use_container_width=True)

            # Manual Feature Selection
            st.write("#### Manual Feature Selection")
            final_features = st.multiselect("Select final features for the model:",
                                           selected_features, default=selected_features)
            st.session_state.selected_features = final_features

            st.write(f"**Selected {len(final_features)} features:** {final_features}")

            if st.button("Proceed to Data Split →", key="proceed_split"):
                st.session_state.step = 6
                st.rerun()

        if st.button("← Go Back", key="back_step5"):
            st.session_state.step = 4
            st.rerun()

    # Step 6: Data Split
    elif st.session_state.step == 6:
        st.markdown("<h2 class='step-header'>Step 6: Data Split</h2>", unsafe_allow_html=True)

        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            features = st.session_state.selected_features
            target = st.session_state.target_column

            st.subheader("📊 Train-Test Split Configuration")
            test_size = st.slider("Test Size (%):", 10, 40, 20)
            random_state = st.number_input("Random State:", value=42)

            st.write(f"**Training Set:** {100-test_size}% | **Test Set:** {test_size}%")

            # Prepare X and y
            X = data[features].select_dtypes(include=[np.number])
            y = data[target]

            # Handle categorical target for classification
            if st.session_state.problem_type == "Classification" and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Handle missing values
            X = X.fillna(X.mean())

            if st.button("Split Data"):
                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=random_state
                )

                st.success("✅ Data split successfully!")
                st.write(f"**Training Set:** {st.session_state.X_train.shape[0]} samples")
                st.write(f"**Test Set:** {st.session_state.X_test.shape[0]} samples")

                # Visualize split
                fig = px.pie(values=[st.session_state.X_train.shape[0], st.session_state.X_test.shape[0]],
                           names=['Training', 'Test'], title="Data Split Distribution",
                           color_discrete_sequence=['#1f77b4', '#e74c3c'])
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.X_train is not None:
                if st.button("Proceed to Model Selection →", key="proceed_model"):
                    st.session_state.step = 7
                    st.rerun()

        if st.button("← Go Back", key="back_step6"):
            st.session_state.step = 5
            st.rerun()

    # Step 7: Model Selection
    elif st.session_state.step == 7:
        st.markdown("<h2 class='step-header'>Step 7: Model Selection</h2>", unsafe_allow_html=True)

        problem_type = st.session_state.problem_type

        st.subheader("🤖 Choose Your Model")

        if problem_type == "Regression":
            models = ["Linear Regression", "SVR (Support Vector Regression)", "Random Forest Regressor"]
        else:
            models = ["Logistic Regression", "SVC (Support Vector Classifier)", "Random Forest Classifier", "K-Means"]

        selected_model = st.selectbox("Select a model:", models)
        st.session_state.selected_model = selected_model

        # Model-specific hyperparameters
        st.subheader("⚙️ Model Hyperparameters")

        if "SV" in selected_model:
            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"])
            st.session_state.kernel = kernel
            C = st.slider("C (Regularization):", 0.1, 10.0, 1.0)
            st.session_state.C = C
            if kernel in ["rbf", "poly", "sigmoid"]:
                gamma = st.selectbox("Gamma:", ["scale", "auto"])
                st.session_state.gamma = gamma

        if "Random Forest" in selected_model:
            n_estimators = st.slider("Number of Estimators:", 10, 200, 100)
            st.session_state.n_estimators = n_estimators
            max_depth = st.slider("Max Depth:", 1, 20, 10)
            st.session_state.max_depth = max_depth

        if selected_model == "K-Means":
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
            st.session_state.n_clusters = n_clusters

        st.write(f"**Selected Model:** {selected_model}")

        if st.button("Proceed to Training →", key="proceed_train"):
            st.session_state.step = 8
            st.rerun()

        if st.button("← Go Back", key="back_step7"):
            st.session_state.step = 6
            st.rerun()

    # Step 8: Training & Validation
    elif st.session_state.step == 8:
        st.markdown("<h2 class='step-header'>Step 8: Model Training & K-Fold Validation</h2>", unsafe_allow_html=True)

        if st.session_state.X_train is not None:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test

            st.subheader("🔄 K-Fold Cross Validation")
            k_folds = st.slider("Number of Folds:", 2, 10, 5)

            if st.button("Train Model"):
                with st.spinner("Training in progress..."):
                    model = None

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    selected_model = st.session_state.selected_model

                    if selected_model == "Linear Regression":
                        model = LinearRegression()
                    elif selected_model == "SVR (Support Vector Regression)":
                        model = SVR(kernel=st.session_state.kernel, C=st.session_state.C)
                    elif selected_model == "Random Forest Regressor":
                        model = RandomForestRegressor(
                            n_estimators=st.session_state.n_estimators,
                            max_depth=st.session_state.max_depth,
                            random_state=42
                        )
                    elif selected_model == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                    elif selected_model == "SVC (Support Vector Classifier)":
                        model = SVC(kernel=st.session_state.kernel, C=st.session_state.C, probability=True)
                    elif selected_model == "Random Forest Classifier":
                        model = RandomForestClassifier(
                            n_estimators=st.session_state.n_estimators,
                            max_depth=st.session_state.max_depth,
                            random_state=42
                        )
                    elif selected_model == "K-Means":
                        model = KMeans(n_clusters=st.session_state.n_clusters, random_state=42)

                    # Train
                    if selected_model != "K-Means":
                        model.fit(X_train_scaled, y_train)

                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=k_folds, scoring='r2' if st.session_state.problem_type == "Regression" else 'accuracy')

                        st.write("### 📊 Cross-Validation Results")
                        st.write(f"**Mean CV Score:** {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

                        fig = px.box(y=cv_scores, title=f"{k_folds}-Fold Cross-Validation Scores",
                                    labels={'y': 'Score'}, color_discrete_sequence=['#2ecc71'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        model.fit(X_train_scaled)

                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled

                    st.success("✅ Model trained successfully!")

            if st.session_state.model is not None:
                if st.button("Proceed to Performance Metrics →", key="proceed_metrics"):
                    st.session_state.step = 9
                    st.rerun()

        if st.button("← Go Back", key="back_step8"):
            st.session_state.step = 7
            st.rerun()

    # Step 9: Performance Metrics
    elif st.session_state.step == 9:
        st.markdown("<h2 class='step-header'>Step 9: Performance Metrics</h2>", unsafe_allow_html=True)

        if st.session_state.model is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test_scaled
            y_test = st.session_state.y_test
            X_train = st.session_state.X_train_scaled
            y_train = st.session_state.y_train

            selected_model = st.session_state.selected_model
            problem_type = st.session_state.problem_type

            if selected_model != "K-Means":
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                if problem_type == "Regression":
                    # Train metrics
                    st.subheader("📊 Training Set Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mean_absolute_error(y_train, y_pred_train):.4f}")
                    with col2:
                        st.metric("MSE", f"{mean_squared_error(y_train, y_pred_train):.4f}")
                    with col3:
                        st.metric("R² Score", f"{r2_score(y_train, y_pred_train):.4f}")

                    # Test metrics
                    st.subheader("📊 Test Set Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred_test):.4f}")
                    with col2:
                        st.metric("MSE", f"{mean_squared_error(y_test, y_pred_test):.4f}")
                    with col3:
                        st.metric("R² Score", f"{r2_score(y_test, y_pred_test):.4f}")

                    # Overfitting/Underfitting check
                    st.subheader("🔍 Overfitting/Underfitting Analysis")
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)

                    if train_r2 - test_r2 > 0.15:
                        st.warning("⚠️ **Potential Overfitting Detected!** Training R² is significantly higher than Test R².")
                    elif train_r2 < 0.5 and test_r2 < 0.5:
                        st.warning("⚠️ **Potential Underfitting Detected!** Both Training and Test R² are low.")
                    else:
                        st.success("✅ Model appears to be well-fitted!")

                    # Prediction vs Actual plot
                    fig = px.scatter(x=y_test, y=y_pred_test,
                                   title="Predicted vs Actual Values",
                                   labels={'x': 'Actual', 'y': 'Predicted'},
                                   color_discrete_sequence=['#1f77b4'])
                    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                                            mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)

                    # Residuals plot
                    residuals = np.array(y_test) - y_pred_test
                    fig = px.histogram(x=residuals, title="Residuals Distribution",
                                      labels={'x': 'Residuals'}, color_discrete_sequence=['#e74c3c'])
                    st.plotly_chart(fig, use_container_width=True)

                else:  # Classification
                    # Train metrics
                    st.subheader("📊 Training Set Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy_score(y_train, y_pred_train):.4f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y_train, y_pred_train, average='weighted'):.4f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y_train, y_pred_train, average='weighted'):.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1_score(y_train, y_pred_train, average='weighted'):.4f}")

                    # Test metrics
                    st.subheader("📊 Test Set Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_test):.4f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y_test, y_pred_test, average='weighted'):.4f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y_test, y_pred_test, average='weighted'):.4f}")
                    with col4:
                        st.metric("F1 Score", f"{f1_score(y_test, y_pred_test, average='weighted'):.4f}")

                    # Overfitting/Underfitting check
                    st.subheader("🔍 Overfitting/Underfitting Analysis")
                    train_acc = accuracy_score(y_train, y_pred_train)
                    test_acc = accuracy_score(y_test, y_pred_test)

                    if train_acc - test_acc > 0.15:
                        st.warning("⚠️ **Potential Overfitting Detected!** Training accuracy is significantly higher than Test accuracy.")
                    elif train_acc < 0.6 and test_acc < 0.6:
                        st.warning("⚠️ **Potential Underfitting Detected!** Both Training and Test accuracy are low.")
                    else:
                        st.success("✅ Model appears to be well-fitted!")

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred_test)
                    fig = px.imshow(cm, title="Confusion Matrix",
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

                    # Classification Report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred_test, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
            else:
                st.write("K-Means clustering does not have traditional performance metrics.")
                # Show cluster visualization
                st.subheader("Cluster Visualization")
                fig = px.scatter(x=X_test[:, 0], y=X_test[:, 1],
                               color=model.labels_[:len(X_test)],
                               title="Cluster Assignment",
                               labels={'x': 'Feature 1', 'y': 'Feature 2'})
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Proceed to Hyperparameter Tuning →", key="proceed_tune"):
                st.session_state.step = 10
                st.rerun()

        if st.button("← Go Back", key="back_step9"):
            st.session_state.step = 8
            st.rerun()

    # Step 10: Hyperparameter Tuning
    elif st.session_state.step == 10:
        st.markdown("<h2 class='step-header'>Step 10: Hyperparameter Tuning</h2>", unsafe_allow_html=True)

        if st.session_state.model is not None:
            st.subheader("🔧 Tune Model Hyperparameters")

            tuning_method = st.selectbox("Select tuning method:", ["GridSearchCV", "RandomizedSearchCV"])

            selected_model = st.session_state.selected_model
            problem_type = st.session_state.problem_type

            # Define parameter grids
            if selected_model in ["Linear Regression", "Logistic Regression"]:
                st.info("Linear/Logistic Regression has minimal hyperparameters to tune.")

            elif "SV" in selected_model:
                st.write("#### Parameter Grid for SVM")
                param_grid = {
                    'C': st.multiselect("C values:", [0.1, 1, 10, 100], default=[1, 10]),
                    'kernel': st.multiselect("Kernels:", ['rbf', 'linear', 'poly'], default=['rbf', 'linear']),
                    'gamma': st.multiselect("Gamma:", ['scale', 'auto'], default=['scale'])
                }

            elif "Random Forest" in selected_model:
                st.write("#### Parameter Grid for Random Forest")
                param_grid = {
                    'n_estimators': st.multiselect("Number of Estimators:", [50, 100, 150, 200], default=[100, 150]),
                    'max_depth': st.multiselect("Max Depth:", [5, 10, 15, 20, None], default=[10, 20]),
                    'min_samples_split': st.multiselect("Min Samples Split:", [2, 5, 10], default=[2, 5])
                }

            if st.button("Run Hyperparameter Tuning"):
                with st.spinner("Tuning in progress... This may take a while..."):
                    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

                    X_train = st.session_state.X_train_scaled
                    y_train = st.session_state.y_train

                    # Base model
                    if problem_type == "Regression":
                        if selected_model == "SVR (Support Vector Regression)":
                            base_model = SVR()
                            tuned_param_grid = {
                                'C': param_grid['C'],
                                'kernel': param_grid['kernel'],
                                'gamma': param_grid['gamma']
                            }
                        elif selected_model == "Random Forest Regressor":
                            base_model = RandomForestRegressor(random_state=42)
                            tuned_param_grid = {
                                'n_estimators': param_grid['n_estimators'],
                                'max_depth': param_grid['max_depth'],
                                'min_samples_split': param_grid['min_samples_split']
                            }
                    else:
                        if selected_model == "SVC (Support Vector Classifier)":
                            base_model = SVC(probability=True)
                            tuned_param_grid = {
                                'C': param_grid['C'],
                                'kernel': param_grid['kernel'],
                                'gamma': param_grid['gamma']
                            }
                        elif selected_model == "Random Forest Classifier":
                            base_model = RandomForestClassifier(random_state=42)
                            tuned_param_grid = {
                                'n_estimators': param_grid['n_estimators'],
                                'max_depth': param_grid['max_depth'],
                                'min_samples_split': param_grid['min_samples_split']
                            }

                    if tuning_method == "GridSearchCV":
                        search = GridSearchCV(base_model, tuned_param_grid, cv=5, scoring='r2' if problem_type == "Regression" else 'accuracy', n_jobs=-1)
                    else:
                        search = RandomizedSearchCV(base_model, tuned_param_grid, cv=5, n_iter=10, scoring='r2' if problem_type == "Regression" else 'accuracy', n_jobs=-1)

                    search.fit(X_train, y_train)

                    st.success("✅ Hyperparameter tuning completed!")
                    st.write("### 📊 Best Parameters")
                    st.json(search.best_params_)
                    st.write(f"**Best Score:** {search.best_score_:.4f}")

                    # Results comparison
                    results_df = pd.DataFrame(search.cv_results_)
                    st.write("### Top 5 Parameter Combinations")
                    st.dataframe(results_df[['params', 'mean_test_score', 'std_test_score']].head())

                    # Update model with best parameters
                    st.session_state.model = search.best_estimator_

            st.markdown("---")
            st.subheader("🎉 Pipeline Complete!")
            st.balloons()

            if st.button("🔄 Start New Pipeline"):
                st.session_state.step = 1
                st.session_state.data = None
                st.session_state.processed_data = None
                st.session_state.model = None
                st.rerun()

        if st.button("← Go Back", key="back_step10"):
            st.session_state.step = 9
            st.rerun()

if __name__ == "__main__":
    main()