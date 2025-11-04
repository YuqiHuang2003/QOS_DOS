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
from Levenshtein import distance as levenshtein_distance
class generate_agent(ABS_Agent):
    def __init__(self,lm) -> None:
        super().__init__(lm)
        self.logger=logging.getLogger("Generate_Agent")
        config_path=os.path.dirname(__file__)+"/types_defination.json"
        self.type_define=json.load(open(config_path))
        self.ac_types=["Claim","Premise","MajorClaim","Evidence","fact","testimony","value","policy","reference"]
        self.ar_types=["Support","Attack",'reason','evidence','Partial-Attack']
    ins_prompt="""
Your task is to write an argumentative text based on the provided topic.
Topic:
{topic}
The text should be in a JSON format, which consists of a main text (context) with placeholders ([AC1], [AC2], etc.), the argument components (argument_component_info) that will be inserted at these placeholders, and argument relations (argument_relation_info) between the argument components.
The types of the argument components are defined as follows:
{argument_component_defination}


Demo Input:
{demo_topic}
Demo Output:
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
        ar_info=[]
        for edge in demo["edges"]:
            from_id=edge["source"]
            to_id=edge["target"]
            if  '_' in edge["label"]:
                ar_type=edge["label"].split("_")[-1]
            else:
                ar_type=edge["label"]
            ar_info.append([f"AC{from_id}",ar_type,f"AC{to_id}"])
        new_json["argument_component_info"]=ac_info
        new_json["argument_relation_info"]=ar_info
        new_json_str=json.dumps(new_json,indent=4)
        ins=self.ins_prompt.format(demo_topic=demo['topic'],argument_component_defination=type_defination,new_json=new_json_str,topic=topic)
        return ins
    def parse_response(self,response,data):
        try:
            if not isinstance(response,Dict):
                self.logger.warning(f"Not Dict response:{response}")
                return None
            demo=data["demo"]
            if "argument_component_info" not in response:
                self.logger.warning(f"No key 'argument_component_info' in response:{response}")
                return None
            ac_infos=response["argument_component_info"]
            new_text=response["context"]
            
            try:
                for AC_i,AC_i_dir in ac_infos.items():
                    new_text=new_text.replace(AC_i,AC_i_dir["content"])
            except:
                self.logger.warning(f"Error in {AC_i},new_text:{new_text}")
                return None
            new_data={"id":data["id"],"input":new_text,"topic":None,"demo_id":demo["id"],"framework": demo["framework"]}
            new_nodes=[]
            if len(ac_infos)>0:
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
                for i,new_node in enumerate(new_nodes):
                    new_node["id"]=i
                    for t in self.ac_types:
                        if levenshtein_distance(new_node["label"],t)<=2:
                            new_node["label"]=t
                            break
                    if new_node["label"] not in self.ac_types:
                        self.logger.warning(f"Error in AC_type:{new_node['label']}")
                        return None
            new_data["nodes"]=new_nodes
            new_edges=[]
            ar_infos=response["argument_relation_info"]
            for ar_info in ar_infos:
                if len(ar_info)!=3:
                    self.logger.warning(f"Error in AR_info:{ar_info}")
                    return None
                if not ar_info[0].startswith("AC") or not ar_info[2].startswith("AC"):
                    self.logger.warning(f"Error in AR_info:{ar_info}")
                    continue
                from_id=ar_info[0].split("AC")[-1]
                to_id=ar_info[2].split("AC")[-1]
                ar_type=ar_info[1]
                for r in self.ar_types:
                    if levenshtein_distance(ar_type,r)<=2:
                        ar_type=r
                        break
                if ar_type not in self.ar_types:
                    self.logger.warning(f"Error in AR_type:{ar_type}")
                    continue
                if from_id==to_id:
                    continue
                new_edges.append({"source":int(from_id),"target":int(to_id),"label":ar_type})
            new_data["edges"]=new_edges
            return new_data
        except Exception as e:
            self.logger.warning(f"Error in parse_response:{e}")
            return None
    def add_dataset_name(self,data,dataset_name):
        if dataset_name.startswith("aaec"):
            return data
        name_map={
            "cdcp":"CDCP",
            "abstrct":"AbstRCT",
            "aaec_essay":""
        }
        prefix=name_map[dataset_name]
        for node in data["nodes"]:
            node['label']=prefix+'_'+node['label']
        for edge in data["edges"]:
            edge['label']=prefix+'_'+edge['label']
        return data
    def generate(self,datas,dataset_name,**kwargs):
        datas=[data for data in datas if data["demo"]["nodes"]]
        ins_list=[self.get_prompt(data,dataset_name) for data in datas]
        responses=self.multi_query(ins_list,json_type=True,not_use_cache=True)
        new_datas=[self.parse_response(response,data) for response,data in zip(responses,datas)]
        new_datas=[self.clear(data) for data in new_datas if data is not None]
        new_datas=[data for data in new_datas if data is not None]
        new_datas=[self.add_dataset_name(data,dataset_name) for data in new_datas]
        return new_datas
if __name__=="__main__":
    pass