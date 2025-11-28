import paramiko
import os
import stat
from typing import Optional, List, Tuple

class SSHManager:
    def __init__(self, hostname, username, key_path=None, password=None, port=22):
        self.hostname = hostname
        self.username = username
        self.key_path = key_path
        self.password = password
        self.port = port
        self.client = None

    def connect(self):
        """Establishes the SSH connection."""
        if self.client:
            return

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            connect_kwargs = {
                "hostname": self.hostname,
                "username": self.username,
                "port": self.port,
            }
            if self.key_path:
                connect_kwargs["key_filename"] = self.key_path
            if self.password:
                connect_kwargs["password"] = self.password

            self.client.connect(**connect_kwargs)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.client = None
            raise e

    def close(self):
        """Closes the SSH connection."""
        if self.client:
            self.client.close()
            self.client = None

    def execute_command(self, command, non_blocking=False):
        """
        Executes a command on the remote server.
        If non_blocking is True, it runs the command in the background and returns immediately.
        """
        if not self.client:
            self.connect()

        if non_blocking:
            # Run in background, redirecting stdout/stderr so it doesn't hang
            # Using nohup and & to detach
            full_command = f"nohup {command} > /dev/null 2>&1 & echo $!"
            stdin, stdout, stderr = self.client.exec_command(full_command)
            pid = stdout.read().decode().strip()
            return pid, ""
        else:
            stdin, stdout, stderr = self.client.exec_command(command)
            # Read output (blocking)
            out = stdout.read().decode()
            err = stderr.read().decode()
            return out, err

    def upload_file(self, local_path, remote_path):
        """Uploads a file via SFTP."""
        if not self.client:
            self.connect()

        sftp = self.client.open_sftp()
        try:
            sftp.put(local_path, remote_path)
        finally:
            sftp.close()

    def download_file(self, remote_path, local_path):
        """Downloads a file via SFTP."""
        if not self.client:
            self.connect()

        sftp = self.client.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()

    def list_files(self, remote_path) -> List[str]:
        """Lists files in a remote directory."""
        if not self.client:
            self.connect()

        sftp = self.client.open_sftp()
        try:
            return sftp.listdir(remote_path)
        except FileNotFoundError:
            return []
        finally:
            sftp.close()

    def file_exists(self, remote_path) -> bool:
        if not self.client:
            self.connect()
        sftp = self.client.open_sftp()
        try:
            sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False
        finally:
            sftp.close()
