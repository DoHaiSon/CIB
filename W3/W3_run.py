
import subprocess
import os

with open("log_w3.txt", "w") as log_w3:
        subprocess.Popen('python3 W3.py --job_name "worker" --task_index 2', shell = True, stderr=log_w3)
