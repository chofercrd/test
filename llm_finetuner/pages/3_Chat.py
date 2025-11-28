import streamlit as st
import os
import sys
import shlex

# Add parent dir to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ssh_manager import SSHManager

st.set_page_config(page_title="Chat", layout="wide")
st.title("Chat with Model")

if "host" not in st.session_state:
    st.warning("Please configure connection in the Training tab first.")
    st.stop()

host = st.session_state.get("host")
user = st.session_state.get("user")
password = st.session_state.get("password")
key_path = st.session_state.get("key_path")
remote_work_dir = st.session_state.get("remote_work_dir", "/tmp/llm_finetuner")

# --- Model Selection ---
if st.button("Refresh Models"):
    try:
        ssh = SSHManager(host, user, key_path=key_path, password=password)
        remote_outputs_dir = f"{remote_work_dir}/outputs"
        # Listing directories in outputs as potential models
        models = ssh.list_files(remote_outputs_dir)
        st.session_state["available_models"] = models
        ssh.close()
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")

model_options = st.session_state.get("available_models", [])
selected_model = st.selectbox("Select Fine-Tuned Model", model_options)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Execute remote inference
        try:
            ssh = SSHManager(host, user, key_path=key_path, password=password)

            # Upload Inference Script if needed (or always to be safe)
            local_inference_script = os.path.join(os.path.dirname(__file__), '../templates/remote_inference.py')
            remote_inference_script = f"{remote_work_dir}/inference.py"
            # Ensure remote dir exists
            ssh.execute_command(f"mkdir -p {remote_work_dir}")
            ssh.upload_file(local_inference_script, remote_inference_script)

            # Securely quote arguments
            safe_model_path = shlex.quote(f"{remote_work_dir}/outputs/{selected_model}")
            safe_prompt = shlex.quote(prompt)
            safe_remote_script = shlex.quote(remote_inference_script)

            # Construct command
            cmd = (f"python3 {safe_remote_script} "
                   f"--model_path {safe_model_path} "
                   f"--prompt {safe_prompt}")

            # Show a spinner
            with st.spinner("Generating response..."):
                stdout, stderr = ssh.execute_command(cmd)

            if stderr and not stdout:
                full_response = f"Error: {stderr}"
            else:
                full_response = stdout.strip()

            ssh.close()
        except Exception as e:
            full_response = f"Connection Error: {e}"

        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
