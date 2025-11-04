import json
import random
from torch.utils.data import Dataset
def read_jsonl(path:str)->list:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f]

class Topic_Demo_Dataset(Dataset):
    def __init__(self,demo_set:list,topic_set:list):
        demos=[]
        topics=[]
        random.shuffle(topic_set)
        total_num=len(demo_set)*12
        while len(topics)<total_num:
            topics+=topic_set
        topics=topics[:total_num]
        while len(demos)<total_num:
            demos+=demo_set
        demos=demos[:total_num]
        self.datas=[{"id":f"id_{id}","demo":demo,"topic":topic} for id,demo,topic in zip(range(total_num),demos,topics)]
    def __len__(self):
        return len(self.datas)
    def __getitem__(self,idx):
        return self.datas[idx]
    @classmethod
    def init(cls,datas:list):
        new_dataset=cls([],[])
        new_dataset.datas=datas
        return new_dataset

    def save_to_jsonl(self,path:str):
        with open(path,"w",encoding="utf-8") as f:
            for item in self.datas:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")
    @classmethod
    def load_from_jsonl(cls,path:str):
        datas=read_jsonl(path)
        new_dataset=cls.init(datas)
        return new_dataset


