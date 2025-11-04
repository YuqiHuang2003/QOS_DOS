try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_html_text
    from .eda_utils import random_eda
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_html_text
    from eda_utils import random_eda
import logging
from typing import Dict
import json
import os
import copy

class eda_agent(ABS_Agent):
    def __init__(self,lm,**kwargs):
        super().__init__(lm)
        self.logger=logging.getLogger("EDA_Agent")
    def eda_call(self,text):
        text=text.strip()
        try:
            if len(text.split(' '))>1:
                text_list=random_eda(text)
                return text_list[0]
        except Exception as e:
            print(f'Error: {{{text}}}')
            input('Wait')
            return text
        return text
    def eda(self,data):
        demo=data["demo"]

        new_data=copy.deepcopy(demo) 
        ori_text=demo["input"]
        ori_start=0
        new_text=''
        for node in new_data["nodes"]:
            
            ori_from,ori_to=node["anchors"][0]["from"],node["anchors"][0]["to"]
            ori_prefix=ori_text[ori_start:ori_from]
            
            ori_node_text=ori_text[ori_from:ori_to]
            #print(ori_start,ori_from,ori_to)
            new_prefix=self.eda_call(ori_prefix)
            new_node_text=self.eda_call(ori_node_text)
            node["anchors"][0]={"from":len(new_text)+len(new_prefix),"to":len(new_text)+len(new_prefix)+len(new_node_text)}
            new_text+=new_prefix
            new_text+=new_node_text
            ori_start=ori_to
        remain_text=ori_text[ori_start:]
        remain_text=self.eda_call(remain_text)
        new_text+=remain_text
        new_data["input"]=new_text
        new_data["demo_id"]=demo["id"]
        return new_data
    



    def generate(self,datas,**kwargs):
        new_datas=[self.eda(data) for data in datas]
        return new_datas
if __name__=="__main__":
    pass