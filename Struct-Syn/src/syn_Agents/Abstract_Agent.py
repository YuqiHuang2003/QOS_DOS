
from typing import List, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import copy
import json
class ABS_Agent():
    def __init__(self,lm) -> None:
        self.lm=lm

    def query(self,query:str,not_use_cache:bool=False)->str:
        response:str =self.lm.query(query,not_use_cache=not_use_cache)
        assert isinstance(response,str)
        return self.lm.remove_markdown(response)
    
    def query_json(self,query:str,not_use_cache:bool=False)->Dict:
        response:str =self.lm.query(query,json_type=True,not_use_cache=not_use_cache)
        while not response.startswith("{"):
            response=response[1:]
        while not response.endswith("}"):
            response=response[:-1]
        return json.loads(response)
    
    def multi_query(self,queries:List[str],json_type:bool=False,not_use_cache:bool=False)->List[str]:
        query_method=self.query_json if json_type else self.query
        responses = [None] * len(queries)
        try:
            with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                futures_to_index = {executor.submit(query_method, query,not_use_cache): i for i, query in enumerate(queries)}
            for future in as_completed(futures_to_index.keys()):
                index = futures_to_index[future]
                try:
                    result = future.result()
                    responses[index] = result
                except Exception as e:
                    logging.error(f"Error occurred in multi_query at index {index}: {e}")
        except Exception as e:
            logging.error(f"Error occurred in multi_query: {e}", exc_info=True)
            raise e
            responses = [self.query(query) for query in queries]
        
        # 确保所有位置都有响应
        for i, response in enumerate(responses):
            if response is None:
                responses[i] = self.query(queries[i])
                
        assert len(queries) == len(responses)
        return responses
    
    def clear(self,data):
        try:
            new_data=copy.deepcopy(data)
            to_delete=[key for key in new_data.keys() if key.endswith("_score")]
            for key in to_delete:
                del new_data[key]
            new_data["nodes"].sort(key=lambda x:x["anchors"][0]["from"])
            for i in range(len(new_data["nodes"])-1):
                if new_data["nodes"][i]["anchors"][0]["to"]>=new_data["nodes"][i+1]["anchors"][0]["from"]:
                    return None
            new_data["nodes"]=[node for node in new_data["nodes"] if node["anchors"][0]["from"]<node["anchors"][0]["to"]]
            for i,node in enumerate(new_data["nodes"]):
                node["id"]=i
            max_id=len(new_data["nodes"])-1
            for edge in new_data["edges"]:
                if edge["source"]>max_id or edge["target"]>max_id:
                    return None
            new_data["edges"].sort(key=lambda x:x["source"]*100+x["target"])
            childs=[edge["target"] for edge in new_data["edges"]]
            new_data["tops"]=[id for id in range(len(new_data["nodes"])) if id not in childs]
            new_data["tops"].sort()
            return new_data
        except Exception as e:
            return None
    