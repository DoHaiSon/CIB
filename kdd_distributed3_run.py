import subprocess
import os

if os.name == 'nt': # Windows
    with open("log_ps0.txt", "w") as log_ps0, open("log_w0.txt", "w") as log_w0, open("log_w1.txt", "w") as log_w1, open("log_w2.txt", "w") as log_w2:
        subprocess.Popen('python kdd_distributed3.py --job_name "ps" --task_index 0', shell = True, stderr=log_ps0)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 0', shell = True, stderr=log_w0)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 1', shell = True, stderr=log_w1)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 2', shell = True, stderr=log_w2)
else:   #Linux
    with open("log_ps0.txt", "w") as log_ps0, open("log_w0.txt", "w") as log_w0, open("log_w1.txt", "w") as log_w1, open("log_w2.txt", "w") as log_w2:
        subprocess.Popen('python3 kdd_distributed3.py --job_name "ps" --task_index 0', shell = True, stderr=log_ps0)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 0', shell = True, stderr=log_w0)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 1', shell = True, stderr=log_w1)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 2', shell = True, stderr=log_w2)
        kill = input('Stop? (y/Y): ')
        if kill == 'y' or kill == 'Y':
               subprocess.Popen('pkill -9 -f python3', shell = True)