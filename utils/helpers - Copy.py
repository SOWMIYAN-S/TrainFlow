import pandas as pd
import numpy as np
import joblib
import io
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def get_algorithms():
    return {
        "Linear Regression": LinearRegression,
        "Random Forest Regressor": RandomForestRegressor,
        "KNN Regressor": KNeighborsRegressor,
        "SVR": SVR,
        "Logistic Regression": LogisticRegression,
        "Decision Tree": DecisionTreeClassifier,
        "Random Forest Classifier": RandomForestClassifier,
        "KNN Classifier": KNeighborsClassifier,
        "SVM": SVC,
        "Naive Bayes": GaussianNB
    }


def preprocess_data(df):
    st.markdown("#### Handle Missing Values")
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
    else:
        impute_cols = st.multiselect("Impute columns with missing values", df.columns)
        if impute_cols:
            strategy = st.selectbox("Imputation strategy", ["mean", "median"])
            imp = SimpleImputer(strategy=strategy)
            df[impute_cols] = imp.fit_transform(df[impute_cols])

    st.markdown("#### Normalize & Transform")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    string_cols = df.select_dtypes(include=['object']).columns.tolist()

    normalize_cols = st.multiselect("Normalize columns", numeric_cols)
    if normalize_cols:
        scaler = StandardScaler()
        df[normalize_cols] = scaler.fit_transform(df[normalize_cols])

    for col in numeric_cols:
        if st.checkbox(f"Make values in '{col}' absolute"):
            df[col] = df[col].abs()

    st.markdown("#### Encode Categorical Columns")
    for col in string_cols:
        if st.checkbox(f"Label Encode '{col}'"):
            try:
                df[col] = LabelEncoder().fit_transform(df[col])
            except:
                st.warning(f"Could not encode {col}")

    st.markdown("#### DateTime Conversion")
    for col in string_cols:
        if st.checkbox(f"Convert '{col}' to datetime"):
            try:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df.drop(columns=[col], inplace=True)
            except:
                st.warning(f"Could not convert {col} to datetime")

    return df


def get_model_instance(X, y, selected_algo, test_size):
    model_class = get_algorithms()[selected_algo]
    model = model_class()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    task_type = "regression" if selected_algo in ["Linear Regression", "Random Forest Regressor", "KNN Regressor", "SVR"] else "classification"
    return model, task_type, X_test, y_test, y_pred


def evaluate_model(task_type, y_test, y_pred):
    if task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        st.success(f"Mean Squared Error: {mse:.4f}")
    else:
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {acc:.4f}")
        st.text(classification_report(y_test, y_pred))


def get_feature_importance(model, x_cols):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=x_cols, ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)
    else:
        st.warning("Model does not support feature importance.")


def save_model_as_zip(model, x_cols, y_col, task_type):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        zipf.writestr("model.pkl", model_bytes.getvalue())

        
        config = f"features = {x_cols}\ntarget = '{y_col}'\nmodel_type = '{task_type}'\n"
        zipf.writestr("model_info.py", config)

    zip_buffer.seek(0)
    return zip_buffer
