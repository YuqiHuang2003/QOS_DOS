try:
    from .Abstract_Agent import ABS_Agent
    from .utils import get_tagged_text,get_ACs
except:
    from Abstract_Agent import ABS_Agent
    from utils import get_tagged_text,get_ACs
import logging
import json
from typing import Dict
class topic_agent(ABS_Agent):
    def __init__(self,lm) -> None:
        super().__init__(lm)
        self.logger=logging.getLogger("Topic_Agent")
    ins_prompt="""
Referring to the argumentative text writing topics below, please brainstorm and write {num} diverse topics.
Please adhere to the following rules:
- Return the results in a similar JSON format.
- Ensure the generated topics cover diverse aspects within the same domain.
{dir}"""
    ins_prompt_aaec="""
Referring to the argumentative text writing topics below, please brainstorm and write {num} diverse topics.
Please adhere to the following rules:
- Return the results in a similar JSON format.
{dir}
"""
    dataset2ins={
        "aaec_essay":ins_prompt_aaec,
        "aaec_para":ins_prompt_aaec,
        "cdcp":ins_prompt,
        "abstrct":ins_prompt
    }
    def get_topic(self,data):
        if "topic" in data:
            return data["topic"]
        else:
            self.logger.warning(f"No topic found in the data: {data['id']}")
            return None
    
    def get_prompt(self,datas,dataset_name):
        num=len(datas)*5
        topics=[]
        for data in datas:
            topic=self.get_topic(data)
            if topic is None:
                continue
            topics.append(topic)
        topic_dir={"topics":topics}
        new_topic_dir=json.dumps(topic_dir,indent=4,ensure_ascii=False)
        prompt=self.dataset2ins[dataset_name].format(num=num,dir=new_topic_dir)
        return prompt
        
    def parse_response(self,response):
        assert isinstance(response,dict)
        assert "topics" in response
        return response["topics"]

    def generate(self,datas,dataset_name,**kwargs):
        prompt=self.get_prompt(datas,dataset_name)
        response=self.query_json(prompt)

        response=self.parse_response(response)
        return response
if __name__=="__main__":
    pass