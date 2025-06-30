import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import streamlit as st
import time
from components.ui import load_dark_theme
import sys, os
import importlib
pages_path = os.path.join(os.path.dirname(__file__), "pages")
sys.path.append(pages_path)




# --------------------------------------------------- #



def resource_path(relative_path):
    try:
        base_path =sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        
    return os.path.join(base_path,relative_path)


load_dark_theme()


if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    st.markdown(
        """
        <style>
        .splash-logo {
            text-align: center;
            margin-top: 100px;
        }
        </style>
        <div class='splash-logo'>
            <h1 style='color:pink; font-size:48px;'>TrainFlow</h1>
            <p style='color:gray; font-size:20px;'>Bound By Code</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(5)
    st.session_state.splash_shown = True
    st.rerun()


st.set_page_config(page_title="Train flow", layout="wide")


def save_model_zip_to_path(model, x_cols, y_col, task_type, save_path):
    try:
        with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            model_buffer.seek(0)
            zf.writestr("model.pkl", model_buffer.read())

            
            info = (
                f"# Auto-generated config\n"
                f"features = {x_cols}\n"
                f"target = '{y_col}'\n"
                f"model_type = '{task_type}'\n"
            )
            zf.writestr("model_info.py", info)

        return True, f"ZIP saved to: {save_path}"
    except Exception as e:
        return False, f"Save failed: {e}"


def run_model_builder():
    st.title("MODEL BUILDER")

    uploaded = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
        st.subheader("Preview")
        st.write(df.head())
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    st.subheader("Preprocessing")
    if st.checkbox("Drop NA"):
        df.dropna(inplace=True)

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()

    normalize = st.multiselect("Normalize columns", num_cols)
    if normalize:
        df[normalize] = StandardScaler().fit_transform(df[normalize])

    for col in num_cols:
        if st.checkbox(f"Abs() for '{col}'"):
            df[col] = df[col].abs()

    for col in str_cols:
        if st.checkbox(f"Convert '{col}' to int"):
            try:
                df[col] = df[col].astype(int)
            except:
                st.warning(f"Can't convert {col} to int")
        if st.checkbox(f"Convert '{col}' to datetime"):
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                st.warning(f"Can't convert {col} to datetime")

    all_cols = df.columns.tolist()
    x_cols = st.multiselect("Select X (features)", all_cols)
    y_col = st.selectbox("Select Y (target)", all_cols)

    if not x_cols or not y_col:
        return

    algo = st.selectbox("Choose algorithm", [
        "Linear Regression", "Random Forest Regressor", "KNN Regressor", "SVR",
        "Logistic Regression", "Decision Tree", "Random Forest Classifier", "KNN Classifier", "SVM", "Naive Bayes"
    ])
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    if st.button("Train Model"):
        try:
            X = df[x_cols].values
            y = df[y_col].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            model = None
            task = None

            if algo == "Linear Regression":
                model = LinearRegression(); task = "regression"
            elif algo == "Random Forest Regressor":
                model = RandomForestRegressor(); task = "regression"
            elif algo == "KNN Regressor":
                model = KNeighborsRegressor(); task = "regression"
            elif algo == "SVR":
                model = SVR(); task = "regression"
            elif algo == "Logistic Regression":
                model = LogisticRegression(); task = "classification"
            elif algo == "Decision Tree":
                model = DecisionTreeClassifier(); task = "classification"
            elif algo == "Random Forest Classifier":
                model = RandomForestClassifier(); task = "classification"
            elif algo == "KNN Classifier":
                model = KNeighborsClassifier(); task = "classification"
            elif algo == "SVM":
                model = SVC(); task = "classification"
            elif algo == "Naive Bayes":
                model = GaussianNB(); task = "classification"

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Model trained")
            if task == "regression":
                st.write("MSE:", mean_squared_error(y_test, y_pred))
            else:
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.text(classification_report(y_test, y_pred))

            # Save in session
            st.session_state.model = model
            st.session_state.x_cols = x_cols
            st.session_state.y_col = y_col
            st.session_state.task_type = task

        except Exception as e:
            st.error(f"Training error: {e}")

    
    if "model" in st.session_state:
        st.subheader("Save Trained Model as ZIP")

        save_path = st.text_input("Enter full file path to save ZIP:", value=os.path.join(os.getcwd(), "trained_model.zip"))

        if st.button("Save ZIP to File"):
            success, msg = save_model_zip_to_path(
                st.session_state.model,
                st.session_state.x_cols,
                st.session_state.y_col,
                st.session_state.task_type,
                save_path
            )
            if success:
                st.success(msg)
            else:
                st.error(msg)


if __name__ == "__main__":
    run_model_builder()
