import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
import importlib
from components.ui import load_dark_theme
from io import BytesIO
from PIL import Image

from utils.plotting_engine import plot_chart
from utils.chart_exporter import export_chart_as_png, export_chart_as_gif
from utils.style_selector import apply_style
from utils.error_modal import show_error_modal


def resource_path(relative_path):
    try:
        base_path =sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        
    return os.path.join(base_path,relative_path)


load_dark_theme()


def app():
    
    st.set_page_config(page_title="TrainFlow", layout="wide")
    st.title("DATA VISUALIZER")

    if "chart_config" not in st.session_state:
        st.session_state.chart_config = {
            "plot_type": "scatter",
            "x_col": None,
            "y_col": None,
            "hue_col": None,
            "title": "",
            "x_label": "",
            "y_label": "",
            "color": "#36C",
            "alpha": 0.8,
            "rotation": 0,
            "grid": True,
            "style": "whitegrid",
            "backend": "Seaborn"
        }

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if not uploaded_file:
        st.info("Please upload a CSV file to get started.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully.")
        st.dataframe(df.head())
    except Exception as e:
        show_error_modal("File Load Failed", str(e))
        st.stop()

    st.subheader("🎛Chart Configuration")

    cfg = st.session_state.chart_config

    cfg["style"] = st.selectbox("Select chart style", ["darkgrid", "whitegrid", "white", "ticks", "dark"], index=1)
    apply_style(cfg["style"])

    cfg["backend"] = st.radio("Plotting Backend", ["Seaborn", "Matplotlib"], horizontal=True)

    plot_types = ["scatter", "line", "bar", "hist", "box", "violin", "heatmap", "pairplot",
                  "kde", "area", "pie", "count", "lm", "strip", "swarm", "3d_scatter"]
    cfg["plot_type"] = st.selectbox("Plot Type", plot_types, index=0)

    # Columns
    cfg["x_col"] = st.selectbox("X-axis", ["None"] + list(df.columns), index=1 if cfg["x_col"] is None else df.columns.get_loc(cfg["x_col"]) + 1)
    cfg["y_col"] = st.selectbox("Y-axis", ["None"] + list(df.columns), index=2 if cfg["y_col"] is None else df.columns.get_loc(cfg["y_col"]) + 1)
    cfg["hue_col"] = st.selectbox("Group/Color By", ["None"] + list(df.columns), index=0)

    multi_cols = []
    if cfg["plot_type"] in ["pairplot", "heatmap"]:
        multi_cols = st.multiselect("Select columns for multi-variate plots", df.columns.tolist(),
                                    default=df.select_dtypes(include='number').columns.tolist())

    st.subheader("Chart Preview")
    preview_map = {
        "scatter": "Points comparison across axes", "line": "Continuous line for trends",
        "bar": "Categorical height visual", "hist": "Distribution of values",
        "box": "Distribution + outliers", "violin": "Smooth distribution shape",
        "heatmap": "Correlation matrix heat", "pairplot": "Multi-variate relationships",
        "kde": "Smoothed density curve", "area": "Area under line curve",
        "pie": "Categorical portions", "count": "Count frequency",
        "lm": "Regression with confidence", "strip": "Category point spread",
        "swarm": "Dense categorical points", "3d_scatter": "3D points visualization"
    }
    st.info(preview_map.get(cfg["plot_type"], "No preview available."))

    with st.expander("Style Customizations"):
        cfg["title"] = st.text_input("Chart Title", value=cfg["title"] or f"{cfg['plot_type'].capitalize()} Chart")
        cfg["x_label"] = st.text_input("X-axis Label", value=cfg["x_col"])
        cfg["y_label"] = st.text_input("Y-axis Label", value=cfg["y_col"])
        cfg["color"] = st.color_picker("Color", value=cfg["color"])
        cfg["alpha"] = st.slider("Opacity", 0.0, 1.0, value=cfg["alpha"])
        cfg["rotation"] = st.slider("X Rotation", 0, 90, value=cfg["rotation"])
        cfg["grid"] = st.checkbox("Show Grid", value=cfg["grid"])

    st.subheader("Chart Output")
    try:
        fig = plot_chart(
            df=df,
            plot_type=cfg["plot_type"],
            x_col=None if cfg["x_col"] == "None" else cfg["x_col"],
            y_col=None if cfg["y_col"] == "None" else cfg["y_col"],
            hue_col=None if cfg["hue_col"] == "None" else cfg["hue_col"],
            multi_cols=multi_cols,
            style=cfg["style"],
            alpha=cfg["alpha"],
            color=cfg["color"],
            rotation=cfg["rotation"],
            grid=cfg["grid"],
            title=cfg["title"],
            x_label=cfg["x_label"],
            y_label=cfg["y_label"],
            backend=cfg["backend"]
        )

        st.pyplot(fig)

    except Exception as e:
        show_error_modal("Chart Render Failed", str(e))

    st.subheader("Export Chart")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download PNG", data=export_chart_as_png(fig), file_name="chart.png", mime="image/png")

    if cfg["plot_type"] == "animated_line":
        with col2:
            gif_data = export_chart_as_gif(df, cfg["x_col"], cfg["y_col"], cfg["color"])
            st.download_button("Download GIF", data=gif_data, file_name="animated_chart.gif", mime="image/gif")

# Run when imported or executed directly
if __name__ == "__main__" or True:
    app()
