import subprocess
import os

with open("log_ps0.txt", "w") as log_ps0:
        subprocess.Popen('python3 Server.py --job_name "ps" --task_index 0', shell = True, stderr=log_ps0)
