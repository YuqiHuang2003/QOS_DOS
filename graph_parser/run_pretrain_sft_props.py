import time
import numpy as np
import os, sys
import argparse
import datetime
from os.path import join
import pandas as pd
import json
import csv
import subprocess  # 添加这个导入
import signal
import atexit
import psutil
running_processes = []
output_dir = ""
def cleanup():
    print("\n执行清理，终止所有子进程...")
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
            print(f"已发送终止信号给进程: {child.pid}")
        except psutil.NoSuchProcess:
            continue
    gone, alive = psutil.wait_procs(children, timeout=5)
    for process in alive:
        try:
            process.kill()
            print(f"强制终止进程: {process.pid}")
        except psutil.NoSuchProcess:
            continue
    try:
        for dir in os.listdir(output_dir):
            if dir.startswith("prop_") and os.path.isdir(join(output_dir,dir)):
                try:
                    os.system("python ./utils/result.py  --dir "+join(output_dir,dir))
                except Exception as e:
                        pass
                    #print(e)
    except Exception as e:
        pass
        #print(e)
    print("所有子进程已终止，清理完成。")
def signal_handler(signum, frame):
    print(f"\n捕获到信号 {signum}，准备退出...")
    sys.exit(0)  # 触发atexit清理
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTSTP, signal_handler)  # Ctrl+Z
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
def read_jsonl(file):
    datas = []
    with open(file, 'r') as f:
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas
def save_jsonl(datas, output_file):
    with open(output_file, "w") as fout:
        for data in datas:
            if data:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')	
def split_data(file,GT_file,output_path,props):
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    datas=read_jsonl(file)
    GT_datas=read_jsonl(GT_file)
    lens=int(len(GT_datas)*props)
    datas=datas[:lens]

    save_jsonl(datas,output_path)


parser = argparse.ArgumentParser()

parser.add_argument("--few_shot", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--agent", type=str)
# parser.add_argument("--start_seed", type=int, default=0)
parser.add_argument("--gpu_id", nargs='+',type=int)
parsed_args = parser.parse_args()

map_id = {0:0, 1:1, 2:2, 3:3}


few_shot = parsed_args.few_shot
dataset_name = parsed_args.dataset
agent = parsed_args.agent
target_gpu_id = parsed_args.gpu_id
print(target_gpu_id)
assert dataset_name in ["cdcp", "abstrct","aaec_para","aaec_essay"]
assert few_shot in [100, 5,]

if "imitation" in agent or 'baseline' in agent:
    original_train1_path=f"../Data/{dataset_name}/syn_datas/{agent}/distill_{agent}_{few_shot}percent.jsonl"
else:
    original_train1_path=f"../Data/{dataset_name}/syn_datas/{agent}/{agent}_{few_shot}percent.jsonl"

log_dir="./log"
model_path="PATH_to_longformer-base"

log_dir=log_dir+f"/{dataset_name}"
syn_dataset =f"{ original_train1_path.split('/')[-1].split('.')[0]}"
output_dir = f"{log_dir}/syn_pretrain_sft/{syn_dataset}/{dataset_name}_{few_shot}percent"
now_time = datetime.datetime.now()
now_time = now_time.strftime("%Y-%m-%d-%X-%a")
output_dir = join(output_dir, now_time)
sleep_time=60
config_file = os.path.join(output_dir, 'config.json')
os.makedirs(output_dir, exist_ok=True)
with open(config_file, 'w') as f:
    f.write(json.dumps({'original_train1_path': original_train1_path, 
                        'few_shot': few_shot,
                        'model_path': model_path},
                        indent=4))
dirs=[]
train2_path = f"../data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_train.jsonl"

running_processes = []

total_epochs=35 if few_shot==100 else 20
prop_list=[2]

for prop_num in prop_list:
    prop_path=output_dir+f'/prop_{prop_num}'
    train1_path=prop_path+f'/pretrain.jsonl'
    split_data(original_train1_path,train2_path,train1_path,prop_num)
    for seed in [42,0,1,2,3]:
        sft_output_dir = join(prop_path, f"seed{seed}")
        while True:
            tmp_path=log_dir+"/tmp_"
            os.system(f'nvidia-smi -q -d Memory |grep -A7 GPU|grep Free > {tmp_path}')
            memory_gpu = [int(x.split()[2]) for x in open(tmp_path, 'r').readlines()]
            os.system(f'rm {tmp_path}')
            available_gpus = [gpu_id for gpu_id, mem in enumerate(memory_gpu) if mem > 29000]
            available_gpus=[t_id for t_id in target_gpu_id if t_id in available_gpus]
            print(f"可用GPU: {available_gpus}")


            if available_gpus:  # 如果有可用GPU
                gpu_id = available_gpus[0]  # 选择第一个可用的GPU
                command = f"python  run_iter_step_pretrain.py " \
                            f"--gpu_id {gpu_id} " \
                            f"--start_seed {seed} " \
                            f"--train1_file {train1_path} " \
                            f"--train2_file {train2_path} " \
                            f"--valid_file ../data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_vali.jsonl  " \
                            f"--test_file ../data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_test.jsonl  " \
                            f"--log_dir {sft_output_dir} " \
                            f"--epoch {total_epochs} " \
                            f"--model_path {model_path} " \
                            f"--dataset {dataset_name} " \
                            f"--few_shot {few_shot} " \
                

                os.makedirs(sft_output_dir, exist_ok=True)
                print(command)
                process = subprocess.Popen(command,shell=True,stdout=open(f"{sft_output_dir}/nohup.out", 'w'),          
                                           stderr=subprocess.STDOUT,start_new_session=True)
                running_processes.append((process, sft_output_dir))
                time.sleep(sleep_time)
                break
            time.sleep(sleep_time)
# 修改预训练的等待逻辑
print("等待所有预训练任务完成...")
for process, sft_output_dir in running_processes:
    print(f"等待任务完成: {sft_output_dir}")
    process.wait()  # 等待进程完成
    print(f"任务已完成: {sft_output_dir}")
