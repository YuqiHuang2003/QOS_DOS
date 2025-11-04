from src.LM import GPT
from src.syn_Agents import topic_summary_agent
try:
    from .utils import read_jsonl,Topic_Demo_Dataset
except:
    from utils import read_jsonl,Topic_Demo_Dataset
import random
from torch.utils.data import DataLoader
import os
import logging
import json
from tqdm import tqdm
import argparse
import string
import re
def summary_topic(args):
    output_dir="./log"
    batch_size=8
    target_percent=args.target_percent
    lm_name="gpt-4o"
    dataset_name=args.dataset_name
    target_agent=topic_summary_agent


    output_dir=output_dir+f"/{dataset_name}/topic_summary_agent"
    output_file=f"{output_dir}/topic_summary_agent_{target_percent}percent.jsonl"
    os.makedirs(output_dir,exist_ok=True)
    logging.basicConfig(
                        filename=f"{output_dir}/log.log",
                        filemode="a",
                        format="[%(asctime)s] %(name)s-%(levelname)s-%(message)s",
                        level=logging.INFO,
                        )
    lm=GPT(cache=True,model_name= lm_name)
    lm.set_cache_log(f"{output_dir}/lm_cache_log.json")
    ac_agent=target_agent(lm)

    demo_file=f"../data_mrp/{dataset_name}_{target_percent}percent/{dataset_name}_train.jsonl"
    demo_set=read_jsonl(demo_file)
    try:
        exist_num=len(read_jsonl(output_file))
    except:
        exist_num=0
    total_num=len(demo_set)-exist_num
    demo_set=demo_set[exist_num:]

    batch=[]
    for data in tqdm(demo_set):
        batch.append(data)
        if len(batch)==batch_size:
            new_batch=ac_agent.generate(datas=batch,dataset_name=dataset_name)
            with open(output_file,"a+") as f:
                for data in new_batch:
                    f.write(json.dumps(data,ensure_ascii=False)+"\n")
            batch=[]
            #break
    if batch:
        new_batch=ac_agent.generate(datas=batch,dataset_name=dataset_name)
        with open(output_file,"a+") as f:
            for data in new_batch:
                f.write(json.dumps(data,ensure_ascii=False)+"\n")

if __name__=="__main__":
    for dataset_name in ["aaec_essay","cdcp","abstrct"]:
        for target_percent in [5,100]:
            args=argparse.Namespace()
            args.dataset_name=dataset_name
            args.target_percent=target_percent
            summary_topic(args)
