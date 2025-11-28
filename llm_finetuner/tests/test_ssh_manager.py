import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ssh_manager import SSHManager

class TestSSHManager(unittest.TestCase):
    @patch('paramiko.SSHClient')
    def test_connect(self, mock_ssh_client):
        manager = SSHManager("example.com", "user", password="password")
        manager.connect()
        mock_ssh_client.return_value.connect.assert_called_with(
            hostname="example.com", username="user", port=22, password="password"
        )

    @patch('paramiko.SSHClient')
    def test_execute_command(self, mock_ssh_client):
        manager = SSHManager("example.com", "user", password="password")
        mock_client_instance = mock_ssh_client.return_value

        # Mock stdout/stderr
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"output"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""

        mock_client_instance.exec_command.return_value = (None, mock_stdout, mock_stderr)

        out, err = manager.execute_command("ls")
        self.assertEqual(out, "output")
        self.assertEqual(err, "")

if __name__ == '__main__':
    unittest.main()
