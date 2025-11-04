from src.LM import GPT
from src.syn_Agents import topic_agent
try:
    from .utils import read_jsonl
except:
    from utils import read_jsonl
import random
import os
import logging
import json
from tqdm import tqdm
import argparse
import string
import re
def generate_topic(args):
    output_dir="./log"
    batch_size=8
    target_percent=args.target_percent
    lm_name="gpt-4o-high-temp"
    dataset_name=args.dataset_name
    target_agent=topic_agent
    total_num=args.total_num
    
    output_dir=output_dir+f"/{dataset_name}/topic_agent"
    output_file=f"{output_dir}/topic_agent_{target_percent}percent.jsonl"
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
    demo_file=f"./log/{dataset_name}/topic_summary_agent/topic_summary_agent_{target_percent}percent.jsonl"
    demo_set=read_jsonl(demo_file)
    batches=[]
    
    try:
        exist_num=len(read_jsonl(output_file))
    except:
        exist_num=0
    total_num=int(total_num-exist_num/5)
    while len(batches)*batch_size<total_num:
        random.shuffle(demo_set)
        for i in range(len(demo_set)//batch_size):
            batches.append(demo_set[i*batch_size:(i+1)*batch_size])
    for batch in tqdm(batches,desc=f"Generating topics for {dataset_name} {target_percent}%"):
        try:
            new_batch=ac_agent.generate(datas=batch,dataset_name=dataset_name)
        except Exception as e:
            logging.error(f"Error in generating topics for {dataset_name} {target_percent}: {e}",exc_info=True)
            continue
        if not new_batch:
            print("No new data")
            continue
        with open(output_file,"a") as f:
            for data in new_batch:
                f.write(json.dumps(data,ensure_ascii=False)+"\n")
def not_allow(s):
    if len(s)<10 or len(s)>300:
        return True
    allowed_punctuation = string.punctuation 
    pattern = r'[^a-zA-Z\s' + re.escape(allowed_punctuation) + ']'
    return bool(re.search(pattern, s))
def filter_topic(args):
    from sentence_transformers import SentenceTransformer,util
    import torch
    from transformers import AutoTokenizer,AutoModel
    class str2embed:
        def __init__(self,init_texts:list):
            path="path_to_sup-simcse-roberta-large"
            self.model=SentenceTransformer(path)
            self.model.eval()
            self.maps={}
            self.batch_size=32
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.add_text(init_texts)
            
        def add_text(self,texts:list):
            if isinstance(texts,str):
                texts=[texts]
            # 找出未缓存的文本
            new_texts = [text for text in texts if text not in self.maps]
            
            # 批量处理新文本
            if new_texts:
                with torch.no_grad():
                    # 分批处理以避免内存不足
                    for i in range(0, len(new_texts), self.batch_size):
                        batch = new_texts[i:i + self.batch_size]
                        embeddings = self.model.encode(
                            batch, 
                            convert_to_tensor=True,
                            device=self.device
                        )
                        # 将批处理结果存入映射
                        for text, emb in zip(batch, embeddings):
                            self.maps[text] = emb
        def embed(self,text):
            if text not in self.maps:
                self.maps[text]=self.model.encode(text, convert_to_tensor=True)
            return self.maps[text]
        def simlarity(self,text1,text2):
            return util.cos_sim(self.embed(text1),self.embed(text2))
    
    
    target_percent=args.target_percent
    dataset_name=args.dataset_name
    demo_file=f"./log/{dataset_name}/topic_summary_agent/topic_summary_agent_{target_percent}percent.jsonl"
    topic_file=f"./log/{dataset_name}/topic_agent/topic_agent_{target_percent}percent.jsonl"
    output_file=f"./log/{dataset_name}/topic_agent/filtered_topic_agent_{target_percent}percent.jsonl"
    demo_set=read_jsonl(demo_file)
    topic_set=open(topic_file,"r").readlines()
    topic_set=[topic.strip().strip('"') for topic in topic_set]
    topic_set=list(set(topic_set))
    topic_set=[topic for topic in topic_set if not not_allow(topic)]
    os.system(f"rm {output_file}")
    demo_topics=[demo["topic"] for demo in demo_set]
    maps=str2embed(demo_topics)
    maps.add_text(topic_set)
    filtered_topics_set=[]
    for topic in tqdm(topic_set):
        max_sim=0
        max_sim_topic=None
        if topic in demo_topics or topic in filtered_topics_set:
            continue
        assert topic not in demo_topics and topic not in filtered_topics_set
        for demo_topic in demo_topics:
            
            sim=maps.simlarity(topic,demo_topic)
            if sim>max_sim:
                max_sim=sim
                max_sim_topic=demo_topic
            if max_sim>args.bar:
                    break
        
        if max_sim<args.bar:
            #print(topic,max_sim,max_sim_topic)
            demo_topics.append(topic)
            filtered_topics_set.append(topic)
            with open(output_file,"a") as f:
                    f.write(topic+"\n")

if __name__=="__main__":
    for dataset_name in ["cdcp","abstrct","aaec_essay"]:#["aaec_essay",]
        for target_percent in [5,100]:
            args=argparse.Namespace()
            args.dataset_name=dataset_name
            args.target_percent=target_percent
            args.bar=0.8
            args.total_num=750 if target_percent==100 else 50
            
            try:
                logging.info(f"Generating topics for {dataset_name} {target_percent}")
                generate_topic(args)
                filter_topic(args)
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                logging.error(f"Error in {dataset_name} {target_percent}: {e}",exc_info=True)
                continue
            
            
            output_dir="./log"
            dataset_name=args.dataset_name
            target_percent=args.target_percent
            filtered_file=f"{output_dir}/{dataset_name}/topic_agent/filtered_topic_agent_{target_percent}percent.jsonl"
            try:
                topic_set=open(filtered_file,"r").readlines()
            except:
                continue
            output_file=f"{output_dir}/{dataset_name}/topic_agent/Topics_{target_percent}percent.jsonl"
            with open(output_file,"w") as f:
                for topic in topic_set:
                    f.write(json.dumps({"topic":topic.strip()},ensure_ascii=False)+"\n")