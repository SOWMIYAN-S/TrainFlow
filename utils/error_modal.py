
import streamlit as st

def show_error_modal(title: str, message: str):
    st.error(f"? **{title}**\n\n{message}")
    with st.expander("?? Troubleshooting Suggestions"):
        if "x_col" in message or "y_col" in message:
            st.write("- Check if your selected X or Y columns contain numeric values.")
        if "heatmap" in message:
            st.write("- Heatmaps work only with numeric data. Remove non-numeric columns.")
        if "3d" in message:
            st.write("- 3D plots require three numeric columns.")
        if "animated" in message:
            st.write("- Animated charts need valid time or ordered data for x-axis.")
        st.write("- Reload the dataset or restart the app if problem persists.")
