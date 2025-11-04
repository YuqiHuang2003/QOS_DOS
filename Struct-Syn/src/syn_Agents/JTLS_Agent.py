try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_tagged_text,get_ACs,get_html_text
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_tagged_text,get_ACs,get_html_text
import logging
from typing import Dict
import json
import os
import copy
class jtls_agent(ABS_Agent):
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
The text should be in a JSON format, which consists of a main text (context) with tags (<AC1>,</AC1>,etc.).
relations and type should be consistent with the argument component defination or demo.
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
        tagged_demos=get_html_text(demo,use_label=False)
        new_json={"context":tagged_demos}
        id2type_map={}
        for node in demo["nodes"]:
            label=node["label"]
            if '_' in label:
                label=label.split('_')[-1]
            id2type_map[node["id"]]=label
        ar_info=[]
        for edge in demo["edges"]:
            from_id=edge["source"]
            to_id=edge["target"]
            label=edge["label"]
            if '_' in label:
                label=label.split('_')[-1]
            r_dir={
                f"AC{from_id}":{"type":id2type_map[from_id]},
                f"AC{to_id}":{"type":id2type_map[to_id]},
                "relation":label
            }
            ar_info.append(r_dir)
        new_json["argument_structure_info"]=ar_info
        new_json_str=json.dumps(new_json,indent=4)
        ins=self.ins_prompt.format(demo_topic=demo['topic'],argument_component_defination=type_defination,new_json=new_json_str,topic=topic)
        return ins
    def parse_response(self,response,data):
        try:
            if not isinstance(response,Dict):
                self.logger.warning(f"Not Dict response:{response}")
                return None
            demo=data["demo"]
            if "argument_structure_info" not in response:
                self.logger.warning(f"No key 'argument_structure_info' in response:{response}")
                return None
            structure_infos=response["argument_structure_info"]
            new_text=response["context"]
            new_nodes=[]
            for ac_id in range(len(structure_infos)*2):
                ac_tag1=f"<AC{ac_id}>"
                ac_tag2=f"</AC{ac_id}>"
                if ac_tag1 not in new_text:
                    break
                ac_from=new_text.index(ac_tag1)
                new_text=new_text[0:ac_from]+new_text[ac_from+len(ac_tag1):]
                ac_to=new_text.index(ac_tag2)
                new_text=new_text[0:ac_to]+new_text[ac_to+len(ac_tag2):]
                ac_dir={
                    "id":ac_id,
                    "label":None,
                    "anchors":[{"from":ac_from,"to":ac_to}]
                }
                new_nodes.append(ac_dir)
            new_edges=[]
            
            for structure_info in structure_infos:
                edge_dir={}
                for k,v in structure_info.items():
                    if k=="relation":
                        edge_dir["label"]=v
                        continue
                    id=int(k[2:])
                    if "source" not in edge_dir.keys():
                        edge_dir["source"]=id
                    else:
                        edge_dir["target"]=id
                    new_nodes[id]["label"]=v["type"]
                new_edges.append(edge_dir)
            new_data={"id":data["id"],"input":new_text,"topic":None,"demo_id":demo["id"],"framework": demo["framework"]}
            for edge in new_edges:
                if "label" not in edge.keys() or "source" not in edge.keys() or "target" not in edge.keys():
                    new_edges.remove(edge)
            for node in new_nodes:
                if not node["label"]:
                    node["label"]="None"
            new_data["input"]=new_text
            new_data["nodes"]=new_nodes
            new_data["edges"]=new_edges
            new_data["tops"]=[]
            
            return new_data
        except Exception as e:
            self.logger.error(f"Error in parse_response:{e}")
            return None
    def add_dataset_name(self,data,dataset_name):
        if dataset_name.startswith("aaec"):
            return data
        name_map={
            "cdcp":"CDCP_",
            "abstrct":"AbstRCT_",
            "aaec_essay":""
        }
        prefix=name_map[dataset_name]
        for node in data["nodes"]:
            if not node["label"].startswith(prefix):
                node['label']=prefix+node['label']
        for edge in data["edges"]:
            if not edge["label"].startswith(prefix):
                edge['label']=prefix+edge['label']
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
    data={"id": "id_0", "demo": {"id": "24001635", "input": " Patients experience reductions in quality of life (QOL) while receiving cancer treatment and several approaches have been proposed to address QOL issues. In this project, the QOL differences between older adult (age 65+) and younger adult (age 18-64) advanced cancer patients in response to a multidisciplinary intervention designed to improve QOL were examined. This study was registered on ClinicalTrials.gov. Newly diagnosed advanced cancer patients undergoing radiation therapy were randomized to active QOL intervention or control groups. Those in the intervention group received six multidisciplinary 90-minute sessions designed to address the five major domains of QOL. Outcomes measured at baseline and weeks 4, 27, and 52 included QOL (Linear Analogue Self-Assessment (LASA), Functional Assessment of Cancer Therapy-General (FACT-G)) and mood (Profile of Mood States (POMS)). Kruskall-Wallis methodology was used to compare scores between older and younger adult patients randomized to the intervention. Of 131 patients in the larger randomized controlled study, we report data on 54 evaluable patients (16 older adults and 38 younger adults) randomized to the intervention. Older adult patients reported better overall QOL (LASA 74.4 vs. 62.9, p = 0.040), higher social well-being (FACT-G 91.1 vs. 83.3, p = 0.045), and fewer problems with anger (POMS anger-hostility 95.0 vs. 86.4, p = 0.028). Long-term benefits for older patients were seen in the anger-hostility scale at week 27 (92.2 vs. 84.2, p = 0.027) and week 52 (96.3 vs. 85.9, p = 0.005). Older adult patients who received a multidisciplinary intervention to improve QOL while undergoing advanced cancer treatments benefited differently in some QOL domains, compared to younger adult patients. Future studies can provide further insight on how to tailor QOL interventions for these age groups. ", "framework": "abstrct", "time": "2020-08-05", "flavor": 0, "version": 1.0, "language": "en", "provenance": "https://gitlab.com/tomaye/abstrct/", "source": "https://gitlab.com/tomaye/abstrct/", "nodes": [{"id": 0, "label": "Evidence", "anchors": [{"from": 1185, "to": 1405}]}, {"id": 1, "label": "Evidence", "anchors": [{"from": 1406, "to": 1560}]}, {"id": 2, "label": "Claim", "anchors": [{"from": 1561, "to": 1765}]}], "edges": [{"source": 2, "target": 0, "label": "AbstRCT_Support"}, {"source": 2, "target": 1, "label": "AbstRCT_Support"}], "tops": [2], "topic": "Older adult cancer patients experience better quality of life improvements than younger patients from multidisciplinary interventions during treatment."}, "topic": {"topic": "Efficacy of parp inhibitors as maintenance therapy in maximizing progression-free survival in BRCA-mutated ovarian cancer."}}
    agent=jtls_agent(None)
    ins=agent.get_prompt(data,"abstrct")
    print(ins)
    pass