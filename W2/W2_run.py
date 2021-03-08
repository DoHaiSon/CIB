
import subprocess
import os

with open("log_w2.txt", "w") as log_w2:
        subprocess.Popen('python3 W2.py --job_name "worker" --task_index 1', shell = True, stderr=log_w2)
