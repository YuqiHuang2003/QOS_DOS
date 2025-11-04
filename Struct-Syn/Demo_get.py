try:
    from .utils import read_jsonl,Topic_Demo_Dataset
except:
    from utils import read_jsonl,Topic_Demo_Dataset
import os
def get_demo(dataset_name,target_percent):
    demo_path=f"./log/{dataset_name}/topic_summary_agent/topic_summary_agent_{target_percent}percent.jsonl"
    topic_path=f"./log/{dataset_name}/topic_agent/Topics_{target_percent}percent.jsonl"
    topic_set=read_jsonl(topic_path)
    demo_set=read_jsonl(demo_path)
    print(len(demo_set))
    output_path=f"./log/{dataset_name}/topic_demo_dataset/topic_demo_dataset_{target_percent}percent.jsonl"
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    dataset=Topic_Demo_Dataset(demo_set,topic_set)
    dataset.save_to_jsonl(output_path)
if __name__=="__main__":
    for dataset_name in ["cdcp","abstrct","aaec_essay"]:
        for target_percent in [5,100]:
            get_demo(dataset_name,target_percent)