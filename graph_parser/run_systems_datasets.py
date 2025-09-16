import json
import time
import numpy as np
import os, sys
import argparse
import datetime
from os.path import join
import csv
import pandas as pd
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=-1)
parser.add_argument("--few_shot", type=int, default=5)
parser.add_argument("--dataset", type=str, default="cdcp")
parsed_args = parser.parse_args()

gpu_id = parsed_args.gpu_id
map_id = {0:0, 1:1, 2:2, 3:3}

few_shot = parsed_args.few_shot
dataset_name = parsed_args.dataset
train_path = f"./Struct-Syn/data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_train.jsonl"

now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = "./log"
log_dir=log_dir+f"/{dataset_name}"
output_dir =  f"{log_dir}/train/{dataset_name}_{few_shot}percent/{now_time}/"
model_path="PATH_to_longformer-base"
os.makedirs(output_dir, exist_ok=True)

save_results_file = os.path.join(output_dir, 'results_all.csv')
csv_exists = os.path.isfile(save_results_file)
dataset_lr={"aaec_para":9.1e-5, "aaec_essay":9.1e-5, "cdcp":5.6e-5, "abstrct":8.1e-5}
dataset_lambda={"aaec_para":(1,0.18,1.05,0.21), "aaec_essay":(1,0.18,1.05,0.21), "cdcp":(1,0.057,0.82,0.15), "abstrct":(1,0.035,0.58,0.17)}
dataset_epoch={"aaec_para":20, "aaec_essay":20, "cdcp":20, "abstrct":20}
sleep_time=60

running_processes=[]
dirs=[]
for seed in [42,0, 1, 2,3]:
    for epoch in [dataset_epoch[dataset_name]]:
        for batch_size in [4]:
            for lr in [
                       dataset_lr[dataset_name]
                       ]:
                for lambda_bio,lambda_proposition,lambda_arc,lambda_rel in [
                    dataset_lambda[dataset_name]
            ]:
            
                    sft_output_dir = join(output_dir, f"seed_{seed}_lr_{lr}_lambda_{lambda_bio}_{lambda_proposition}_{lambda_arc}_{lambda_rel}/sft")
                    dirs.append(sft_output_dir)

                    while True:
                        tmp_path=log_dir+"/tmp_"
                        os.system(f'nvidia-smi -q -d Memory |grep -A7 GPU|grep Free > {tmp_path}')
                        memory_gpu = [int(x.split()[2]) for x in open(tmp_path, 'r').readlines()]
                        os.system(f'rm {tmp_path}')
                        
                        # 检查所有可用GPU
                        available_gpus = [gpu_id for gpu_id, mem in enumerate(memory_gpu) if mem > 29000]
                        print(f"可用GPU: {available_gpus}")
                        
                        
                        if available_gpus:  # 如果有可用GPU
                            gpu_id = available_gpus[0]  # 选择第一个可用的GPU
                            command = f"CUDA_VISIBLE_DEVICES={gpu_id} python -m amparse.trainer.train " \
                                        f"--log {sft_output_dir} " \
                                        f"--ftrain {train_path} " \
                                        f"--fvalid ./Struct-Syn/data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_vali.jsonl  " \
                                        f"--ftest ./Struct-Syn/data_mrp/{dataset_name}_{few_shot}percent/{dataset_name}_test.jsonl  " \
                                        f"--seed {seed} " \
                                        f"--model_name_or_path {model_path} " \
                                        f"--build_numericalizer_on_entire_corpus true " \
                                        f"--batch_size 4 " \
                                        f"--eval_batch_size 16 " \
                                        f"--embed_dropout 0.1 " \
                                        f"--mlp_dropout 0.1 " \
                                        f"--dim_mlp 768 " \
                                        f"--dim_biaffine 768 " \
                                        f"--lambda_bio {lambda_bio} " \
                                        f"--lambda_proposition {lambda_proposition} " \
                                        f"--lambda_arc {lambda_arc} " \
                                        f"--lambda_rel {lambda_rel} " \
                                        f"--lr {lr} " \
                                        f"--warmup_ratio 0.1 " \
                                        f"--clip 5.0 " \
                                        f"--epochs {epoch} " \
                                        f"--terminate_epochs {epoch} " \
                                        f"--evaluate_epochs 1 "
                            
                            print(f"启动任务在GPU {gpu_id}: {command}")
                            os.makedirs(sft_output_dir, exist_ok=True)

                            process = subprocess.Popen(
                                command,
                                shell=True,
                                stdout=open(f"{sft_output_dir}/nohup.out", 'w'),
                                stderr=subprocess.STDOUT,
                                start_new_session=True
                            )
                            running_processes.append((process, sft_output_dir))
                            time.sleep(sleep_time)
                            break
                        time.sleep(sleep_time)

# 修改预训练的等待逻辑
print("等待所有预训练任务完成...")
for process, pretrain_output_dir in running_processes:
    print(f"等待任务完成: {pretrain_output_dir}")
    process.wait()  # 等待进程完成
    print(f"任务已完成: {pretrain_output_dir}")


