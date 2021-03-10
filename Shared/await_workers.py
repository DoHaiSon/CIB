import asyncio
import pathlib
import os
import pysftp

logs_flag_W1 = "/home/avitech-pc/haison98/CIB/kdd_ddl3-2/flags"
log_flag_W2 = "/home/avitech/haison98/CIB/kdd_ddl3-2/flags"
worker_user = ["avitech-pc", "avitech", "avitech"]
worker_pass = ["1", "1", "1"]

async def waitting():
    await asyncio.sleep(0.1) 
    print("waiting another workers") 

def await_another_workers(W, worker, LOG_DIR, epoch):
    log_W1 = ""
    log_W2 = ""
    #log_W3 = ""
    log_W1_dir = LOG_DIR + "/flags/flag_W1"
    log_W2_dir = LOG_DIR + "/flags/flag_W2"
    #log_W3_dir = LOG_DIR + "/flags/flag_W3"
    if W == 0:                  ## Check flag of anthors worker, if all are true, next Epoch, else await
        send_flag(W, worker, epoch)
        while not str(epoch) in log_W2: #and "true" in log_W3:
            waitting()
            with open(log_W2_dir, 'r') as flag_W2: #, open(log_W3_dir, 'r') as flag_W3:
                log_W2 = flag_W2.read()
                #log_W3 = flag_W3.read()    
        #os.remove(LOG_DIR + "/flags/flag_W2")
        #os.remove(LOG_DIR + "/flags/flag_W3")

    elif W == 1:                ## Create a flag file with true value in server PC after done an Epoch
        send_flag(W, worker, epoch)
        while not str(epoch) in log_W1:
            waitting()
            with open(log_W1_dir, 'r') as flag_W1:
                log_W1 = flag_W1.read()
    else:                       ## Create a flag file with true value in server PC after done an Epoch
        send_flag(W, worker, epoch)

def delete_folder(pth) :
    try:
        for sub in pth.iterdir() :
            if sub.is_dir() :
                delete_folder(sub)
            else :
                sub.unlink() 
        print("Clean flag files.")       
    except:
        print("The flags folder does not exist.")

def send_flag(W, worker, epoch):
    if W == 0:
        for i in range (1, len(worker)):
            with pysftp.Connection(host=worker[i][:-5], username=worker_user[i], password=worker_pass[i]) as sftp:
                print("Connection succesfully stablished ...")

                # Define the file that you want to upload from your local directorty
                # or absolute "C:\Users\sdkca\Desktop\TUTORIAL2.txt"
                localFilePath = 'flag_W' + str(W+1)
                with open(localFilePath, 'w') as flag_file:
                    flag_file.write(str(epoch))

                # Define the remote path where the file will be uploaded
                remoteFilePath = logs_flag_W2 + "/" + localFilePath

                sftp.put(localFilePath, remoteFilePath)
                print("Sent flags epoch: {} to anther workers.".format(epoch))
    else:
        with pysftp.Connection(host=worker[0][:-5], username=worker_user[0], password=worker_pass[0]) as sftp:
            print("Connection succesfully stablished ...")

            # Define the file that you want to upload from your local directorty
            # or absolute "C:\Users\sdkca\Desktop\TUTORIAL2.txt"
            localFilePath = 'flag_W' + str(W+1)
            with open(localFilePath, 'w') as flag_file:
                flag_file.write(str(epoch))

            # Define the remote path where the file will be uploaded
            remoteFilePath = logs_flag_W1 + "/" + localFilePath

            sftp.put(localFilePath, remoteFilePath)
            print("Sent flags epoch: {} to server.".format(epoch))
