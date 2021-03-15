import asyncio
import pathlib
import os
import pysftp
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

list_worker = ['worker_0', 'worker_1', 'worker_2']
server_user = ["avitech-pc"]
server_pass = ["1"]

async def waitting():
    await asyncio.sleep(50) 

def await_another_workers(W, worker, logs_flag, layer, epoch):
    ## Check flag of anthors worker, if all are true, next Epoch, else await
    flag = True
    send_flag(W, worker, logs_flag, layer, epoch)
    while flag:
        waitting()
        flag = read_log(W, logs_flag, layer, epoch)

def send_flag(W, worker, logs_flag, layer, epoch):
    if W == 0:
        with open (logs_flag, "a") as logs_flag:
            logs_flag.write(str(W) + "," + str(layer) + "," + str(epoch) + "\n")
    else:
        with pysftp.Connection(host=worker[0][:-5], username=server_user, password=server_pass) as sftp:
            print("Connection succesfully stablished ...")

            # Define the file that you want to upload from your local or absolute directorty
            localFilePath = "logs_flag"
            sftp.get(logs_flag, localFilePath)
            with open(localFilePath, 'a') as logs_flag:
                logs_flag.write(str(W) + "," + str(layer) + "," + str(epoch) + "\n")

            # Define the remote path where the file will be uploaded
            sftp.put(localFilePath, logs_flag)
            print("Sent flags epoch: {} to server.".format(epoch))

def read_log(W, logs_flag, layer, epoch):
    log = []
    with open (logs_flag, "r") as logs_flag:
        for line in logs_flag:
            log.append(line)
    log_worker = list(map(int, [i.split(',', 2)[0] for i in log] ))
    log_layer = list(map(int, [i.split(',', 2)[1] for i in log] ))
    log_epoch = [i.split(',', 2)[2] for i in log]
    log_epoch = list(map(int, [ x[:-1] for x in log_epoch ] ))
    print(log_worker)
    print(log_layer)
    print(log_epoch)
    his = len(log_worker)
    max_layer = [0, 0, 0]
    max_epoch = [0, 0, 0]
    for i in range (his - 1, his - 13, -1):
        if log_worker[i] != W :
            if log_layer[i] > max_layer[log_worker[i]]:
                max_layer[log_worker[i]] = log_layer[i]
                if log_epoch[i] > max_epoch[log_epoch[i]]:
                    max_epoch[log_worker[i]] = log_epoch[i]
            elif log_layer == max_layer[log_worker[i]]:
                if log_epoch[i] > max_epoch[log_epoch[i]]:
                    max_epoch[log_worker[i]] = log_epoch[i]
    print(max_layer)
    print(max_epoch)
    if layer < max(max_layer) or (layer == max(max_layer) and epoch < max(max_epoch)):
        return True
    return False

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