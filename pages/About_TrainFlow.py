import streamlit as st
from PIL import Image
import importlib, os
from components.ui import load_dark_theme


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
    st.title("ABOUT TRANFLOW")

    st.markdown("""
    ### Welcome to **TrainFlow**

    **TrainFlow** is a powerful, zero-code Machine Learning tool designed for developers, students, and AI enthusiasts who want to build, test, and deploy ML models **without writing a single line of code**.

    ---
    ##### Built With:
    - **Developer:** Sowmiyan S
    - **Company:** Bound By Code
    - **Framework:** Python + Streamlit
    - **Packaging Ready:** PyInstaller/auto-py-to-exe compatible
    - **Interface:** Dark-themed, 2025 UI-ready, responsive layout

    ---

    ### Key Features of TrainFlow:

    #### CSV Upload & Smart Preprocessing
    - Auto-handle missing values
    - Normalize and scale numeric data
    - Convert strings to datetime/int
    - Absolute value transformation
    - No-code feature engineering

    #### Model Building (Train Your Flow)
    - Select your X (features) and Y (target)
    - Choose from 10+ ML algorithms:
        - Linear & Logistic Regression
        - Decision Tree (C & R)
        - Random Forest (C & R)
        - SVM, SVR, Naive Bayes
        - KNN Classifier & Regressor
    - Train/test split slider
    - Auto model training with metrics

    #### Test the Model Instantly
    - Manual input for real-time predictions
    - Shows classification or regression output
    - Model performance metrics:
        - Accuracy, MSE, F1, R², etc.

    #### Export Model as ZIP
    - Save model + metadata as `.zip`
    - Compatible with TrainFlow Runner
    - No browser download dependency — file system save supported

    #### Visualizer 
    - Upload CSV and customize visualizations
    - Choose chart type, x/y columns, themes, labels, styles
    - Export charts as images
    - Uses `matplotlib`, `seaborn`

    #### UI/UX Highlights
    - Clean dark theme
    - Animated loaders
    - Styled navigation (Navbar incoming)
    - Error modals and real-time feedback

    ---

    ### Mission:
    > “TrainFlow was built to make Machine Learning **accessible, faster, and smarter** for creators like you. Whether you’re learning, teaching, building prototypes, or deploying — this is your shortcut.”

    ---

    **Developer:** Sowmiyan S  
    **Company:** Bound By Code  
    Year: 2025  
    Version: TrainFlow 1.0
    """)

# This allows it to be used in page switch
if __name__ == "__main__" or True:
    app()
