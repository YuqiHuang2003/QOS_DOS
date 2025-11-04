
from src.LM import GPT
from src.syn_Agents import ac_sub_agent,ac_rewrite_agent,paraphrase_agent,imitation_agent,imitation_wo_topic_agent,imitation_wo_pattern_agent,baseline_imitation_agent,baseline_paraphrase_agent,generate_agent,eda_agent
try:
    from .utils import read_jsonl,Topic_Demo_Dataset
except:
    from utils import read_jsonl,Topic_Demo_Dataset

from torch.utils.data import DataLoader
import os
import logging
import json
from tqdm import tqdm
import argparse
def main(dataset_name,target_percent,target_agent):


    output_dir="./log"
    batch_size=20
    lm_name="gpt-4o"


    output_dir=output_dir+f"/{dataset_name}/{target_agent.__name__}"
    output_file=f"{output_dir}/{target_agent.__name__}_{target_percent}percent.jsonl"
    os.makedirs(output_dir,exist_ok=True)
    logging.basicConfig(
                        filename=f"{output_dir}/log.log",
                        filemode="a",
                        format="[%(asctime)s] %(name)s-%(levelname)s-%(message)s",
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S"
    
                    )
    lm=GPT(cache=True,model_name= lm_name)
    try:
        lm.set_cache_log(f"{output_dir}/lm_cache_log.json")
    except Exception as e:
        pass
    ac_agent=target_agent(lm)

    try:
        datas=read_jsonl(output_file)
        exist_ids=[data["id"] for data in datas]
    except Exception as e:
        open(output_file,"w").close()
        exist_ids=[]

    dataset=Topic_Demo_Dataset.load_from_jsonl(f"./log/{dataset_name}/topic_demo_dataset/topic_demo_dataset_{target_percent}percent.jsonl")
    print(len(dataset))
    total_num=500 if target_percent==100 else 100
    dataset=dataset[:total_num]
    dataset=[data for data in dataset if data["id"] not in exist_ids]
    print(len(dataset))
    dataloader=DataLoader(dataset,batch_size=batch_size,collate_fn=lambda batch:batch)

    for batch in tqdm(dataloader):
        try:
            new_batch=ac_agent.generate(datas=batch,dataset_name=dataset_name)
        except Exception as e:
            raise e
            continue
        if not new_batch:
            print("No new data")
            continue
        with open(output_file,"a") as f:
            for data in new_batch:
                f.write(json.dumps(data)+"\n")

if __name__=="__main__":
    for dataset_name in ["cdcp","abstrct","aaec_essay"]:
        for target_percent in [5,100]:
            for target_agent in [paraphrase_agent,imitation_agent]:
                print(dataset_name,target_percent,target_agent.__name__)
                main(dataset_name,target_percent,target_agent)
                break



