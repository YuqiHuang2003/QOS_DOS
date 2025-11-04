try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_tagged_text,get_ACs
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_tagged_text,get_ACs
import logging
from typing import Dict
import json
import os
class paraphrase_agent(ABS_Agent):
    def __init__(self,lm) -> None:
        super().__init__(lm)
        self.logger=logging.getLogger("Paraphrase_Agent")
        config_path=os.path.dirname(__file__)+"/types_defination.json"
        self.type_define=json.load(open(config_path))
        

    ins_prompt="""
Your task is to paraphrase the provided argumentative text.
The text is given in a JSON format, which consists of a main text (context) with placeholders ([AC1], [AC2], etc.), and the argument components (argument_component_info) that will be inserted at these placeholders.
The types of the argument components are defined as follows:

{argument_component_defination}

Please adhere to the following rules:

- Preserve the original meaning of both the context and argument components
- Enhance expression diversity and language variety
- After paraphrasing, ensure smooth and natural flow when components are reintegrated into the context
- Maintain each argument component's designated type

Below is the provided argumentative text, please return the answer in a similar JSON format.

{new_json}
"""
    def get_prompt(self,data,dataset_name):
        topic=data["topic"]["topic"]
        type_defination=self.type_define[dataset_name]
        demo=data["demo"]
        text=demo["input"]
        tagged_demos=get_tagged_text(demo)
        new_json={"context":tagged_demos}
        ac_info={}
        for node in demo["nodes"]:
            ac_dir={}
            id=node["id"]
            if "_" in node["label"]:
                ac_dir["type"]=node["label"].split("_")[-1]
            else:
                ac_dir["type"]=node["label"]
            ac_dir["content"]=text[node["anchors"][0]["from"]:node["anchors"][0]["to"]]
            ac_info[f"[[AC{id}]]"]=ac_dir
        new_json["argument_component_info"]=ac_info
        new_json_str=json.dumps(new_json,indent=4)
        ins=self.ins_prompt.format(argument_component_defination=type_defination,new_json=new_json_str)
        return ins
    def parse_response(self,response,data):
        if not isinstance(response,Dict):
            self.logger.warning(f"Not Dict response:{response}")
            return None
        demo=data["demo"]
        if "argument_component_info" not in response:
            self.logger.warning(f"No key 'argument_component_info' in response:{response}")
            return None
        ac_infos=response["argument_component_info"]
        if len(ac_infos.values())!=len(demo["nodes"]):
            self.logger.warning(f"Error in ac_infos:{ac_infos},demo_nodes:{demo['nodes']}")
            return None
        new_text=response["context"]
        
        try:
            for AC_i,AC_i_dir in ac_infos.items():
                new_text=new_text.replace(AC_i,AC_i_dir["content"])
        except:
            self.logger.warning(f"Error in {AC_i},new_text:{new_text}")
            return None
        new_data={"id":data["id"],"input":new_text,"topic":None,"demo_id":demo["id"],"framework": demo["framework"]}
        new_nodes=[]
        if len(demo["nodes"])>0:
            for ac_info in ac_infos.values():
                ac_text=ac_info["content"]
                ac_type=ac_info["type"]
                begin=new_text.find(ac_text)
                if begin==-1:
                    self.logger.warning(f"Error in AC_text:{ac_text},new_text:{new_text}")
                    return None
                end=begin+len(ac_text)
                new_nodes.append({"anchors":[{"from":begin,"to":end}],"label":ac_type})
            new_nodes.sort(key=lambda x:x["anchors"][0]["from"])
            for new_node,old_node in zip(new_nodes,demo["nodes"]):
                new_node["id"]=old_node["id"]
                old_label=old_node["label"] if "_" not in old_node["label"] else old_node["label"].split("_")[-1]
                if new_node["label"]!=old_label:
                    self.logger.warning(f"Error in AC_type:{new_node['label']},old_node:{old_label}")
                    return None
                new_node["label"]=old_node["label"]
        new_data["nodes"]=new_nodes
        new_data["edges"]=demo["edges"]
        return new_data
    def generate(self,datas,dataset_name,**kwargs):
        datas=[data for data in datas if data["demo"]["nodes"]]
        ins_list=[self.get_prompt(data,dataset_name) for data in datas]
        responses=self.multi_query(ins_list,json_type=True,not_use_cache=True)
        new_datas=[self.parse_response(response,data) for response,data in zip(responses,datas)]
        new_datas=[self.clear(data) for data in new_datas if data is not None]
        new_datas=[data for data in new_datas if data is not None]
        return new_datas
if __name__=="__main__":
    pass