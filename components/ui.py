import streamlit as st

def load_dark_theme():
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton>button {
        background-color: #ff2c94;
        color: white;
        padding: 0.5em 1em;
        border: none;
        border-radius: 5px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff00c8;
        transform: scale(1.03);
    }

    .stTextInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: #1c1f26;
        color: #ffffff;
        border: 1px solid #8f2fff;
    }

    .stMarkdown h2, .stMarkdown h3 {
        color: #ff2c94;
    }

    .stAlert {
        background-color: #20242f;
        border-left: 5px solid #ff00c8;
    }

    .css-1rs6os.edgvbvh3 { 
        background-color: #161a24;
    }

    .css-1v0mbdj.e1f1d6gn1 {
        background-color: #2b006b;
        color: #ffffff;
    }

    .stDownloadButton button {
        background-color: #8f2fff;
        color: white;
    }

    .stDownloadButton button:hover {
        background-color: #ff2c94;
        transform: scale(1.02);
    }

    .stRadio > div {
        background-color: #1f2230;
        padding: 0.5em;
        border-radius: 8px;
    }

    .stSelectbox label, .stTextInput label, stmultiselect {
        color: #ff2c94 !important;
    }
    </style>
    """, unsafe_allow_html=True)
