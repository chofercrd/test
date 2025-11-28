import streamlit as st

st.set_page_config(page_title="LLM Fine-Tuner", layout="wide")

st.title("LLM Fine-Tuning Control Center")

st.markdown("""
Welcome to the LLM Fine-Tuning Manager.
Select a page from the sidebar to begin:

*   **Training:** Configure and launch fine-tuning jobs on your remote server.
*   **Analysis:** Monitor training logs and visualize performance metrics.
*   **Chat:** Interact with your fine-tuned models.
""")

# Sidebar for global configuration (like SSH defaults) could go here
st.sidebar.header("Global Settings")
st.sidebar.info("Configure connection details in the specific pages.")
