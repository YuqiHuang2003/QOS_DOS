try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_tagged_text,get_ACs
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_tagged_text,get_ACs
import logging
import json
from typing import Dict
import os
import copy
class topic_summary_agent(ABS_Agent):
    def __init__(self,lm) -> None:
        super().__init__(lm)
        self.demo=json.load(open(os.path.dirname(__file__)+"/Topic_Summary_Demo.json"))
        self.logger=logging.getLogger("Topic_Summary_Agent")
    ins_prompt="""Your task is to summarize the topic of the following argumentative text:
{text}
## Rules:
- Please refer to the following examples and provide a topic of similar length.
- Return the result in a similar JSON format.
{demos}
"""
    def title_clear(self,graph_dir):
        graph_dir=copy.deepcopy(graph_dir)
        text=graph_dir["input"].strip()
        text=text.replace("\n \n","\n\n").replace("\n  \n","\n\n")
        if "\n\n" not in text:
            graph_dir["topic"]=None
        else:
            ls=text.split("\n\n")
            assert len(ls)<=2 and len(ls)>=1,f"title_clear error: {[text]}"
            title=ls[0]
            graph_dir["topic"]=title
            graph_dir["input"]=ls[-1]
            for node in graph_dir["nodes"]:
                node["anchors"][0]["from"]-=len(title)+2
                node["anchors"][0]["to"]-=len(title)+2
        for node in graph_dir["nodes"]:
            if "_" in node["label"]:
                node["label"]=node["label"].split("_")[-1]
        return graph_dir
    
    def get_prompt(self,data,dataset_name):
        assert data["topic"] is None
        demo=self.demo[dataset_name]
        #print(data)
        text=data["input"]
        ins=self.ins_prompt.format(text=text,demos=demo)
        return ins
        
    def parse_response(self,response,data):
        assert isinstance(response,dict),f"{response},{data}"
        assert "topic" in response
        data["topic"]=response["topic"]
        return data

    def generate(self,datas,dataset_name,**kwargs):
        cleared_datas=[self.title_clear(data) for data in datas]
        results=[data for data in cleared_datas if data["topic"] is not None]
        to_generate=[data for data in cleared_datas if data not in results]
        prompts=[self.get_prompt(data,dataset_name) for data in to_generate]
        if prompts:
            responses=self.multi_query(prompts,json_type=True)
            generated=[self.parse_response(response,data) for response,data in zip(responses,to_generate)]
            results+=generated
        datas_ids=[data["id"] for data in datas]
        results.sort(key=lambda x:datas_ids.index(x["id"]))
        return results
if __name__=="__main__":
    pass