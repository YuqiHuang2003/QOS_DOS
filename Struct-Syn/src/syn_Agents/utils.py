
def visual(data:dict)->str:
    text=data["input"]
    node_list=data["nodes"]
    node_map={node["id"]:node for node in node_list}
    edge_list=data["edges"]
    
    if len(node_list)==0:
        label_prompt="None"
        span_prompt="None"
        edge_prompt="None"
    else:
        label_prompt=""
        for node in node_list:
            node_silce=text[node["anchors"][0]["from"]:node["anchors"][0]["to"]]
            label_prompt+=f"{node['id']}:[# {node_silce} #] is {node['label']}\n"
    
        span_prompt=""
        for node in node_list:
            node_silce=text[node["anchors"][0]["from"]:node["anchors"][0]["to"]]
            span_prompt+=f"[[Span {node['id']}]]: {node_silce}\n"
        
        if len(edge_list)>0:
            edge_prompt=""
            for edge in edge_list:
                edge_prompt+=f"<AC {edge['target']}> {edge['label']} <AC {edge['source']}>\n"
        else:
            edge_prompt="None"

    dic={
        "text":text.strip(),
        "label_prompt":label_prompt.strip(),
        "span_prompt":span_prompt.strip(),
        "ac_prompt":span_prompt.replace("Span ","AC ").strip(),
        "edge_prompt":edge_prompt.strip()
    }
    return dic
def get_tagged_text(data:dict)->str:
    text=data["input"]
    text_list=[]
    remain_text=text
    for i,nodes in enumerate(data["nodes"]):
        tag=f"[[AC{i}]]"
        begin=nodes["anchors"][0]["from"]
        end=nodes["anchors"][0]["to"]
        before=text[-len(remain_text):begin]
        remain_text=text[end:]
        new=[before,tag]
        text_list+=new
    text_list.append(remain_text)
    return "".join(text_list).strip()
def get_html_text(data:dict)->str:
    text=data["input"]
    text_list=[]
    remain_text=text
    for i,node in enumerate(data["nodes"]):
        tag1=f" <{node['label']}> "
        tag2=f" </{node['label']}> "
        begin=node["anchors"][0]["from"]
        end=node["anchors"][0]["to"]
        before=text[-len(remain_text):begin]
        ac_text=text[begin:end]
        remain_text=text[end:]
        new=[before,tag1,ac_text,tag2]
        text_list+=new
    text_list.append(remain_text)
    new_text="".join(text_list).strip()
    new_text=new_text.replace("  "," ")
    return new_text
def get_ACs(data:dict)->list:
    return visual(data)["ac_prompt"]
if __name__=="__main__":
    ss=[" [0] "," [0,1] "," [fuck,0,1] "]
    # for s in ss:
    #     print(s)
    #     print(extract_idx_lists(s))
    data={"id": "essay071_4", "input": "Taken all together, advertisements have its social responsibility and contribute to economic growth. We can not deny all of them. However, if those with exaggerated and fake information could be banned, advertisements would be welcomed by more.", "flavor": 0, "language": "en", "framework": "synthetic", "time": "2025-03-28 (18:22)", "version": 1.0, "provenance": "https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp", "source": "https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp", "tops": [0, 1], "edges": [], "nodes": [{"id": 0, "anchors": [{"from": 20, "to": 99}], "label": "MajorClaim"}, {"id": 1, "anchors": [{"from": 104, "to": 129}], "label": "Claim"}], "nodes_score": [{"id": 2, "anchors": [{"from": 20, "to": 99}], "anchors_score": [0.9999986886978149], "label": "MajorClaim", "label_score": 0.9999488592147827}, {"id": 2, "anchors": [{"from": 104, "to": 129}], "anchors_score": [0.6223462224006653], "label": "Claim", "label_score": 0.9999918937683105}], "edges_score": []}
    print(get_html_text(data))
    print(get_tagged_text(data))