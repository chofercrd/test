import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

# Add parent dir to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ssh_manager import SSHManager

st.set_page_config(page_title="Analysis", layout="wide")
st.title("Training Analysis")

# Ensure we have connection details
if "host" not in st.session_state:
    st.warning("Please configure connection in the Training tab first.")
    st.stop()

host = st.session_state.get("host")
user = st.session_state.get("user")
password = st.session_state.get("password")
key_path = st.session_state.get("key_path")
remote_work_dir = st.session_state.get("remote_work_dir", "/tmp/llm_finetuner")

# --- Sync Controls ---
col1, col2 = st.columns([1, 3])

local_log_dir = "llm_finetuner/data/logs"
if not os.path.exists(local_log_dir):
    os.makedirs(local_log_dir)

with col1:
    st.subheader("Sync Logs")
    if st.button("Fetch Remote Logs"):
        try:
            ssh = SSHManager(host, user, key_path=key_path, password=password)
            # Assuming logs are stored in remote_work_dir/outputs/{job_name}/log.csv
            # We will list directories in outputs
            remote_outputs_dir = f"{remote_work_dir}/outputs"

            # Simple approach: List folders, then try to download log.csv from each
            # Note: This is a simplified "sync".
            # Better approach might be to list all csv files recursively, but let's stick to simple structure

            runs = ssh.list_files(remote_outputs_dir)

            synced_count = 0
            for run in runs:
                remote_log_path = f"{remote_outputs_dir}/{run}/log.csv"
                local_run_dir = f"{local_log_dir}/{run}"
                if not os.path.exists(local_run_dir):
                    os.makedirs(local_run_dir)

                # Check if remote file exists (naive try/except)
                try:
                    ssh.download_file(remote_log_path, f"{local_run_dir}/log.csv")
                    synced_count += 1
                except Exception:
                    # Maybe no log yet or not a directory
                    pass

            ssh.close()
            st.success(f"Synced logs for {synced_count} runs.")

        except Exception as e:
            st.error(f"Sync failed: {e}")

# --- Visualization ---
st.divider()

# List local runs
available_runs = [d for d in os.listdir(local_log_dir) if os.path.isdir(os.path.join(local_log_dir, d))]

if not available_runs:
    st.info("No logs found. Run a job and click 'Fetch Remote Logs'.")
else:
    selected_run = st.selectbox("Select Training Run", available_runs)

    log_file = os.path.join(local_log_dir, selected_run, "log.csv")
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            st.write("### Data Preview")
            st.dataframe(df.head())

            st.write("### Plot Metrics")
            c1, c2 = st.columns(2)
            with c1:
                x_axis = st.selectbox("X Axis", df.columns, index=0)
            with c2:
                y_axis = st.multiselect("Y Axis", df.columns, default=[df.columns[-1]] if len(df.columns) > 0 else [])

            if y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"Metrics for {selected_run}")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.warning("log.csv not found for this run.")
