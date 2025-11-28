import streamlit as st
import os
import sys
import shlex

# Add parent dir to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ssh_manager import SSHManager

st.set_page_config(page_title="Training", layout="wide")

st.title("Start Fine-Tuning Job")

# --- Sidebar: Connection Configuration ---
with st.sidebar:
    st.header("Connection Settings")
    host = st.text_input("Host", value=st.session_state.get("host", ""), key="host_input")
    user = st.text_input("Username", value=st.session_state.get("user", ""), key="user_input")
    auth_method = st.radio("Auth Method", ["Password", "Key File"], index=0)

    password = ""
    key_path = ""
    if auth_method == "Password":
        password = st.text_input("Password", type="password", value=st.session_state.get("password", ""))
    else:
        key_path = st.text_input("Private Key Path", value=st.session_state.get("key_path", ""))

    remote_work_dir = st.text_input("Remote Working Directory", value="/tmp/llm_finetuner", help="Where datasets and logs will be stored on the server")

    if st.button("Test Connection"):
        try:
            ssh = SSHManager(host, user, key_path=key_path if key_path else None, password=password if password else None)
            if ssh.connect():
                st.success(f"Connected to {host}")
                ssh.close()
                # Save to session state
                st.session_state["host"] = host
                st.session_state["user"] = user
                st.session_state["password"] = password
                st.session_state["key_path"] = key_path
                st.session_state["remote_work_dir"] = remote_work_dir
        except Exception as e:
            st.error(f"Connection failed: {e}")

# --- Main Interface ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Configuration")
    model_name = st.selectbox("Base Model", ["meta-llama/Llama-2-7b-hf", "google/gemma-7b", "Qwen/Qwen1.5-7B-Chat", "mistralai/Mistral-7B-v0.1"])

    st.subheader("Hyperparameters")
    epochs = st.number_input("Epochs", min_value=1, value=3)
    learning_rate = st.number_input("Learning Rate", value=2e-4, format="%.5f")
    batch_size = st.number_input("Batch Size", min_value=1, value=4)
    lora_rank = st.number_input("LoRA Rank", min_value=8, value=64)

with col2:
    st.subheader("Dataset Selection")
    dataset_source = st.radio("Dataset Source", ["Upload Local File", "Select Remote File"])

    selected_dataset_path = None

    if dataset_source == "Upload Local File":
        uploaded_file = st.file_uploader("Choose a CSV or JSONL file", type=['csv', 'json', 'jsonl'])
        if uploaded_file:
            # We need to save it temporarily to upload
            if not os.path.exists("llm_finetuner/data"):
                os.makedirs("llm_finetuner/data")
            local_path = os.path.join("llm_finetuner/data", uploaded_file.name)
            with open(local_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"File staged locally at {local_path}")
            selected_dataset_path = local_path # Mark for upload
    else:
        # List remote files
        if st.button("List Remote Files"):
            try:
                ssh = SSHManager(st.session_state.get("host"), st.session_state.get("user"),
                                 key_path=st.session_state.get("key_path"), password=st.session_state.get("password"))
                files = ssh.list_files(remote_work_dir)
                ssh.close()
                st.session_state["remote_files"] = files
            except Exception as e:
                st.error(f"Could not list files: {e}")

        remote_files = st.session_state.get("remote_files", [])
        if remote_files:
            selected_remote_file = st.selectbox("Select Remote File", remote_files)
            selected_dataset_path = selected_remote_file # Already on remote

# --- Launch Control ---
st.divider()
job_name = st.text_input("Job Name (for logging)", value="my_finetune_run")

if st.button("Start Training Job"):
    if not st.session_state.get("host"):
        st.error("Please configure connection first.")
    else:
        status_container = st.empty()
        status_container.info("Initiating...")

        ssh = SSHManager(st.session_state.get("host"), st.session_state.get("user"),
                         key_path=st.session_state.get("key_path"), password=st.session_state.get("password"))

        try:
            # 1. Handle Dataset
            remote_dataset_path = ""
            if dataset_source == "Upload Local File":
                if not selected_dataset_path:
                    st.error("No file uploaded.")
                    st.stop()

                remote_dataset_path = f"{remote_work_dir}/{os.path.basename(selected_dataset_path)}"
                status_container.info(f"Uploading dataset to {remote_dataset_path}...")

                # Ensure remote dir exists
                ssh.execute_command(f"mkdir -p {remote_work_dir}")
                ssh.upload_file(selected_dataset_path, remote_dataset_path)
            else:
                remote_dataset_path = f"{remote_work_dir}/{selected_dataset_path}"

            # 2. Upload Training Script
            local_train_script = os.path.join(os.path.dirname(__file__), '../templates/remote_train.py')
            remote_train_script = f"{remote_work_dir}/train.py"
            status_container.info(f"Uploading training script to {remote_train_script}...")
            ssh.upload_file(local_train_script, remote_train_script)

            # 3. Construct Command (Securely)
            # Use shlex.quote to prevent injection
            safe_model_name = shlex.quote(model_name)
            safe_dataset_path = shlex.quote(remote_dataset_path)
            safe_output_dir = shlex.quote(f"{remote_work_dir}/outputs/{job_name}")
            safe_remote_script = shlex.quote(remote_train_script)

            cmd = (f"python3 {safe_remote_script} "
                   f"--model_name {safe_model_name} "
                   f"--dataset_path {safe_dataset_path} "
                   f"--epochs {epochs} "
                   f"--learning_rate {learning_rate} "
                   f"--batch_size {batch_size} "
                   f"--lora_rank {lora_rank} "
                   f"--output_dir {safe_output_dir}")

            status_container.info(f"Launching command: {cmd}")

            # 4. Execute
            pid, _ = ssh.execute_command(cmd, non_blocking=True)

            st.success(f"Training started! PID: {pid}. Check the Analysis tab later for results.")

        except Exception as e:
            st.error(f"Error starting training: {e}")
        finally:
            ssh.close()
