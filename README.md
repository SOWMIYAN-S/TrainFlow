# TrainFlow: Your Zero-Code ML Studio

<img src="https://github.com/SOWMIYAN-S/TrainFlow/blob/main/logo.png" alt="TrainFlow Logo" width="100" height="100">

**TrainFlow** is a powerful, zero-code Machine Learning tool designed for developers, students, and AI enthusiasts to build, test, and deploy ML models without writing a single line of code.

## 🚀 Features

TrainFlow provides an intuitive interface for end-to-end machine learning workflows, from data preprocessing to model deployment.

### 📊 CSV Upload & Smart Preprocessing
* **Auto-handle missing values:** Clean your data effortlessly.
* **Normalize and scale numeric data:** Prepare your features for optimal model performance.
* **Convert strings to datetime/int:** Transform data types with ease.
* **Absolute value transformation:** Apply mathematical transformations directly.
* **No-code feature engineering:** Get your data ready for modeling without complex scripting.

### 🧠 Model Building (Train Your Flow)
* **Intuitive Feature/Target Selection:** Easily select your X (features) and Y (target) variables.
* **Diverse Algorithm Selection:** Choose from over 10 popular ML algorithms:
    * Linear Regression
    * Logistic Regression
    * Decision Tree (Classifier & Regressor)
    * Random Forest (Classifier & Regressor)
    * Support Vector Machine (SVM)
    * Support Vector Regressor (SVR)
    * K-Nearest Neighbors (KNN Classifier & Regressor)
    * Naive Bayes
* **Customizable Train/Test Split:** Adjust the split ratio with a simple slider.
* **Automated Training & Metrics:** Train your model with a click and get instant performance metrics.
    * Mean Squared Error (MSE) for regression.
    * Accuracy and Classification Report for classification.

### ✨ Instant Model Testing
* **Real-time Predictions:** Input values manually to get instant predictions.
* **Clear Output:** See classification results or regression values immediately.
* **Performance Metrics Display:** View key metrics like Accuracy, MSE, F1-score, R², etc.

### 📦 Model Export & Runner
* **Export as ZIP:** Save your trained model along with its metadata (features, target, model type) as a compact `.zip` file.
* **TrainFlow Runner Compatible:** The exported ZIP file can be directly loaded and used in the `Model_Runner.py` application for quick predictions.
* **Local File System Save:** No browser download dependencies — file system save supported.

### 📈 Data Visualizer
* **Upload CSV and Customize:** Load your datasets and tailor your visualizations.
* **Extensive Chart Types:** Choose from a wide array of plot types:
    * Scatter, Line, Bar, Histogram, Box, Violin
    * Heatmap, Pairplot, KDE, Area, Pie, Count
    * Linear Model (lm), Strip, Swarm, 3D Scatter
* **Customizable Aesthetics:** Control x/y columns, themes, labels, colors, opacity, and more.
* **Export Charts:** Save your visualizations as high-quality PNG images. GIF export available for animated line plots.
* **Backend Flexibility:** Utilizes both `matplotlib` and `seaborn` for diverse plotting capabilities.

### 🎨 UI/UX Highlights
* **Sleek Dark Theme:** A modern and eye-pleasing dark interface.
* **Animated Loaders:** Smooth visual feedback during processes.
* **Responsive Layout:** Adapts to different screen sizes.
* **Real-time Feedback:** Error modals and success messages for a guided experience.

## 🛠️ Built With

* **Developer:** Sowmiyan S
* **Company:** Bound By Code
* **Framework:** Python + Streamlit
* **ML Libraries:** Scikit-learn (Linear Regression, Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, SVM, Naive Bayes), Joblib
* **Data Handling:** Pandas
* **Visualization:** Matplotlib, Seaborn
* **Packaging Ready:** PyInstaller/auto-py-to-exe compatible

## 🎯 Mission

> “TrainFlow was built to make Machine Learning **accessible, faster, and smarter** for creators like you. Whether you’re learning, teaching, building prototypes, or deploying — this is your shortcut.”

## 🚀 Getting Started

To run TrainFlow locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SOWMIYAN-S/TrainFlow.git](https://github.com/SOWMIYAN-S/TrainFlow.git)
    cd TrainFlow
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # You'll need to create this file based on the imports in your code
    ```
    * **Note:** Based on the provided code, your `requirements.txt` would likely include:
        ```
        streamlit
        joblib
        pandas
        scikit-learn
        matplotlib
        seaborn
        Pillow
        ```

4.  **Run the Streamlit applications:**

    * **To start the Whole application:**
        ```bash
        python launch_app.py
        ```


    TrainFlow will open in your default web browser.


## 📞 Contact

**Developer:** Sowmiyan S
**Company:** Bound By Code
**Year:** 2025
**Version:** TrainFlow 1.0

---
