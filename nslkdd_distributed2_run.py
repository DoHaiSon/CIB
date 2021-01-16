import subprocess

subprocess.Popen('python3 nslkdd_distributed2.py --job_name "ps" --task_index 0', shell = True)
subprocess.Popen('python3 nslkdd_distributed2.py --job_name "worker" --task_index 0', shell = True)
subprocess.Popen('python3 nslkdd_distributed2.py --job_name "worker" --task_index 1', shell = True)