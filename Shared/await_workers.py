import asyncio
import pathlib
import os
import pysftp

logs_flag = ""
worker_user = ["avitech", "avitech", "avitech"]
worker_pass = ["1", "1", "1"]

async def await_another_workers(W, worker, LOG_DIR):
    log_W2 = ""
    log_W3 = ""
    if W == 0:                  ## Check flag of anthors worker, if all are true, next Epoch, else await
        while "true" in log_W2 and "true" in log_W3:
            await asyncio.sleep(0.1) 
            with open(LOG_DIR + "/flags/flag_W2", 'r') as flag_W2, open(LOG_DIR + "/flags/flag_W3", 'r') as flag_W3:
                log_W2 = flag_W2.read()
                log_W3 = flag_W3.read()    
        os.remove(LOG_DIR + "/flags/flag_W2")
        os.remove(LOG_DIR + "/flags/flag_W3")

    elif W == 1:                ## Create a flag file with true value in server PC after done an Epoch
        send_flag(W, worker, logs_flag)
    else:                       ## Create a flag file with true value in server PC after done an Epoch
        send_flag(W, worker, logs_flag)

def delete_folder(pth) :
    for sub in pth.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else :
            sub.unlink() 
    print("Clean flag files.")       

def send_flag(W, worker, logs_flag):
    with pysftp.Connection(host=worker[W-1], username=worker_user[0], password=worker_pass[0]) as sftp:
        print("Connection succesfully stablished ...")

        # Define the file that you want to upload from your local directorty
        # or absolute "C:\Users\sdkca\Desktop\TUTORIAL2.txt"
        localFilePath = 'flag_W' + str(W+1)

        # Define the remote path where the file will be uploaded
        remoteFilePath = logs_flag + "/" + localFilePath

        sftp.put(localFilePath, remoteFilePath)