
import subprocess
import os

with open("log_w1.txt", "w") as log_w1:
        subprocess.Popen('python3 W1.py --job_name "worker" --task_index 0', shell = True, stderr=log_w1)
        kill = input('Stop? (y/Y): ')
        if kill == 'y' or kill == 'Y':
               subprocess.Popen('pkill -9 -f python3', shell = True)