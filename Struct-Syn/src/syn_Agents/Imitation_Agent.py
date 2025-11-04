try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_html_text
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_html_text
import logging
from typing import Dict
import json
import os
import copy
class imitation_agent(ABS_Agent):
    def __init__(self,lm) -> None:
        super().__init__(lm)
        self.logger=logging.getLogger("Imitation_Agent")
        config_path=os.path.dirname(__file__)+"/types_defination.json"
        self.type_define=json.load(open(config_path))
        all_types=["Claim","Premise","MajorClaim","Evidence","fact","testimony","value","policy","reference"]
        self.tags=[]
        for t in all_types:
            #用于删除前后多余的空格
            self.tags+=[f" <{t}> ",f" </{t}> ",f" <{t}>",f" </{t}>",f"<{t}> ",f"</{t}> ",f"<{t}>",f"</{t}>"]
    ins_prompt="""
Your task is to imitate the provided reference text to write a new argumentative text.
The topic of the new text should be:
{topic}
Please adhere to the following rules:

- Ensure the number of paragraphs is the same as the reference text, and the text length is similar.
- Make adjustments in aspects such as the organization of the argumentative structure, logical reasoning patterns, or the selection of evidence types.
- In the `argumentation_pattern`, the sequence of argument components describes the logic flow of the entire text. The types and definitions of these components are as follows:
{argument_component_defination}
- First, adjust the provided `argumentation_pattern` to create a *new* `argumentation_pattern`. Then, generate a new argumentative text following this new `argumentation_pattern`. To adjust the `argumentation_pattern`, you can choose to perform one or more of the following operations:
    - Add new argument components.
    - Remove existing argument components.
    - Adjust the order of the argument components.
    - Adjust the type of the argument components.

Below is the reference text, please return the answer in a similar JSON format.

{new_json}
""" 
    def title_clear(self,graph_dir):
        graph_dir=copy.deepcopy(graph_dir)
        text=graph_dir["input"].strip()
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
    def get_para_pattern(self,graph_dir):
        text=graph_dir["input"]
        paragraph_list=text.split("\n")
        para_num=len(paragraph_list)
        para_pattern={f"paragraph_{i}":[] for i in range(para_num)}
        para_end_idx=[]
        for i,para in enumerate(paragraph_list):
            start=para_end_idx[-1] if para_end_idx else 0
            end=start+len(para)+1
            para_end_idx.append(end)
            for node in graph_dir["nodes"]:
                if node["anchors"][0]["from"]>=start and node["anchors"][0]["to"]<=end:
                    para_pattern[f"paragraph_{i}"].append(node["label"])
        for para_i,para_types in para_pattern.items():
            para_pattern[para_i]=" -> ".join(para_types)
        return para_pattern
    def get_prompt(self,data,dataset_name):
        demo=self.title_clear(data["demo"])
        topic=data["topic"]["topic"]
        type_defination=self.type_define[dataset_name]
        html_text=get_html_text(demo)
        json_dir={"topic":demo["topic"],"argumentation_pattern":self.get_para_pattern(demo),"argumentative_text":html_text,}
        ins_prompt=self.ins_prompt.format(new_json=json.dumps(json_dir,indent=4,ensure_ascii=False),argument_component_defination=type_defination,topic=topic)
        return ins_prompt

    def parse_response(self,response,data):
        assert isinstance(response,Dict)
        demo=data["demo"]
        argumentation_pattern=response["argumentation_pattern"]
        topic=response["topic"]
        tagged_text=response["argumentative_text"]
        for tag in self.tags:
            tagged_text=tagged_text.replace(tag,"")
        tagged_text=tagged_text.replace("\n\n","\n")
        new_text=tagged_text
        new_data={"id":data["id"],"input":new_text,"topic":topic,"argumentation_pattern":argumentation_pattern,"demo_id":demo["id"],"framework": demo["framework"]}
        new_nodes=[]
        
        new_data["nodes"]=[]
        new_data["edges"]=[]
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
    data={"id": "id_2", "demo": {"id": "essay089", "input": "Some people think that human needs for farmland, housing and industry are more\n\nNowadays there are some issues which many countries and nations are concern about that. Two of the most important of them are rate of increasing population and endangered plants and animals. From my point of view although endangered animals are important, human needs should be considered more.\nFirstly, day by day population is increasing, so human needs for food is increasing as well. We need farmland to grow crops such as wheat and corn, and there is no doubt that without food people cannot alive for more than some weeks. Therefore, farmlands are vital for human life.\nSecondly, people need shelters to protect themselves from mother nature such as wind and flood and wild animals, so they need land to build their houses on it. Dwelling is one of the most important human needs after food which is undeniable.\nFinally, human need some crucial things to continue their life, and most of them are industrial such as medicine. Therefore, some lands should be assigned to industry. Todays life has some difficulty which become comfortable with the help of industry. \nTo conclude, the population of endangered animals are less than people's population, so they do not need so much land that people need. I would maintain that nowadays humans need farmland, housing and industry because of increasing population are more important than saving land for endangered animals.", "framework": "synthetic", "time": "2020-08-05", "flavor": 0, "version": 1.0, "language": "en", "provenance": "https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp", "source": "https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp", "nodes": [{"id": 0, "label": "Claim", "anchors": [{"from": 302, "to": 334}]}, {"id": 1, "label": "MajorClaim", "anchors": [{"from": 336, "to": 373}]}, {"id": 2, "label": "Premise", "anchors": [{"from": 384, "to": 419}]}, {"id": 3, "label": "Premise", "anchors": [{"from": 424, "to": 466}]}, {"id": 4, "label": "Premise", "anchors": [{"from": 468, "to": 607}]}, {"id": 5, "label": "Claim", "anchors": [{"from": 620, "to": 654}]}, {"id": 6, "label": "Premise", "anchors": [{"from": 666, "to": 767}]}, {"id": 7, "label": "Claim", "anchors": [{"from": 772, "to": 814}]}, {"id": 8, "label": "Premise", "anchors": [{"from": 816, "to": 896}]}, {"id": 9, "label": "Premise", "anchors": [{"from": 907, "to": 1010}]}, {"id": 10, "label": "Claim", "anchors": [{"from": 1023, "to": 1064}]}, {"id": 11, "label": "Premise", "anchors": [{"from": 1066, "to": 1148}]}, {"id": 12, "label": "Premise", "anchors": [{"from": 1164, "to": 1234}]}, {"id": 13, "label": "Claim", "anchors": [{"from": 1239, "to": 1285}]}, {"id": 14, "label": "MajorClaim", "anchors": [{"from": 1309, "to": 1360}]}, {"id": 15, "label": "Claim", "anchors": [{"from": 1372, "to": 1452}]}], "edges": [{"source": 3, "target": 2, "label": "Support"}, {"source": 5, "target": 3, "label": "Support"}, {"source": 5, "target": 4, "label": "Support"}, {"source": 7, "target": 6, "label": "Support"}, {"source": 7, "target": 8, "label": "Support"}, {"source": 10, "target": 9, "label": "Support"}, {"source": 10, "target": 11, "label": "Support"}, {"source": 13, "target": 12, "label": "Support"}], "tops": [0, 1, 5, 7, 10, 13, 14, 15]}, "topic": {"domain": "Politics", "conception": "Political accountability", "topic": "Political accountability is undermined when citizens disengage from the electoral process."}}
    agent=imitation_agent(None)
    print(agent.get_prompt(data,"aaec_essay"))