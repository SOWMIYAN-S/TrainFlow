import streamlit as st
import joblib
import zipfile
import tempfile
import os
import importlib
import importlib.util
import pandas as pd
from components.ui import load_dark_theme
from PIL import Image


def resource_path(relative_path):
    try:
        base_path =sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        
    return os.path.join(base_path,relative_path)


load_dark_theme()


def app():
    load_dark_theme()
    st.set_page_config(page_title="TrainFlow", layout="wide")
    st.title("MODEL RUNNER")

    uploaded_zip = st.file_uploader("Upload your exported model ZIP file", type=["zip"])

    if not uploaded_zip:
        st.info("Upload a ZIP exported from the Model Builder to proceed.")
        st.stop()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getvalue())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            model = joblib.load(os.path.join(tmpdir, "model.pkl"))

            spec = importlib.util.spec_from_file_location("model_info", os.path.join(tmpdir, "model_info.py"))
            model_info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_info)

            features = model_info.features
            target = model_info.target
            model_type = model_info.model_type

            st.success(f"Model Loaded: {model_type.upper()}")

            user_inputs = []
            st.subheader("Enter Input for Prediction")
            for col in features:
                val = st.number_input(f"{col}", format="%.4f", key=f"input_{col}")
                user_inputs.append(val)

            if st.button("Predict"):
                try:
                    prediction = model.predict([user_inputs])[0]
                    st.markdown(f"<h1 style='color:lime;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    except Exception as e:
        st.error(f"Something went wrong loading the ZIP: {e}")


if __name__ == "__main__" or True:
    app()
