
import subprocess
import os

with open("log_w0.txt", "w") as log_w0:
        subprocess.Popen('python3 W1.py --job_name "worker" --task_index 0', shell = True, stderr=log_w0)
