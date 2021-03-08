
import subprocess
import os

with open("log_w1.txt", "w") as log_w1:
        subprocess.Popen('python3 W1.py --job_name "worker" --task_index 0', shell = True, stderr=log_w1)
