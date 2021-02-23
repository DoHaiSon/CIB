import subprocess
import os

if os.name == 'nt': # Windows
    with open("stdout_ps0.txt","w") as out_ps0, open("stderr_ps0.txt", "w") as err_ps0, open("stdout_w0.txt","w") as out_w0, open("stderr_w0.txt", "w") as err_w0, \
    open("stdout_w1.txt","w") as out_w1, open("stderr_w1.txt", "w") as err_w1, open("stdout_w2.txt","w") as out_w2, open("stderr_w2.txt", "w") as err_w2:
        subprocess.Popen('python kdd_distributed3.py --job_name "ps" --task_index 0', shell = True, stdout=out_ps0, stderr=err_ps0)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 0', shell = True, stdout=out_w0, stderr=err_w0)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 1', shell = True, stdout=out_w1, stderr=err_w1)
        subprocess.Popen('python kdd_distributed3.py --job_name "worker" --task_index 2', shell = True, stdout=out_w2, stderr=err_w2)
else:   #Linux
    with open("stdout_ps0.txt","w") as out_ps0, open("stderr_ps0.txt", "w") as err_ps0, open("stdout_w0.txt","w") as out_w0, open("stderr_w0.txt", "w") as err_w0, \
    open("stdout_w1.txt","w") as out_w1, open("stderr_w1.txt", "w") as err_w1, open("stdout_w2.txt","w") as out_w2, open("stderr_w2.txt", "w") as err_w2:
        subprocess.Popen('python3 kdd_distributed3.py --job_name "ps" --task_index 0', shell = True, stdout=out_ps0, stderr=err_ps0)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 0', shell = True, stdout=out_w0, stderr=err_w0)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 1', shell = True, stdout=out_w1, stderr=err_w1)
        subprocess.Popen('python3 kdd_distributed3.py --job_name "worker" --task_index 2', shell = True, stdout=out_w2, stderr=err_w2)

