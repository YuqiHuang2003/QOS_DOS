Exploring Quality and Diversity in Synthetic Data Generation for Argument Mining
======================
EMNLP 2025 Main Conference
Abstract
===
```text
The advancement of Argument Mining (AM)
is hindered by a critical bottleneck: the scarcity
of structure-annotated datasets, which are ex-
pensive to create manually. Inspired by recent
successes in synthetic data generation across
various NLP tasks, this paper explores method-
ologies for LLMs to generate synthetic data for
AM. We investigate two complementary syn-
thesis perspectives: a quality-oriented synthesis
approach, which employs structure-aware para-
phrasing to preserve annotation quality, and a
diversity-oriented synthesis approach, which
generates novel argumentative texts with di-
verse topics and argument structures. Experi-
ments on three datasets show that augmenting
original training data with our synthetic data,
particularly when combining both quality- and
diversity-oriented instances, significantly en-
hances the performance of existing AM mod-
els, both in full-data and low-resource settings.
Moreover, the positive correlation between syn-
thetic data volume and model performance
highlights the scalability of our methods.
```
```text
### Project Structure
QOS+DOS/
    ├── Data/
    │   ├── aaec_essay/syn_datas
    │   │   ├── imitation_agent/ datas(5% and 100% without annotation)
    │   │   └── paraphrase_agent/ datas(5% and 100%)
    │   └── abstrct, cdcp ....
    │           
    ├── data_mrp    #orginal dataset from AAEC,CDCP and AbstRCT 
    │
    ├── graph_parser # ST Model
    │
    └── Struct-Syn #  see /Struct-Syn/README.md
```
Quick Start 
==
1. Evironment 
```bash
pip install -r requirements.txt
```
2. prepare API and URL for LLMs in Struct-Syn/src/LM/config.json

3. using Topic_Summary to get original topic from dataset.Then brainstorm more topics.
```bash
cd Struct-Syn
python Topic_Summary.py
python Topic_Generate.py
```
4. Generate QOS and DOS dataset (adjust agent parameters and workflow as needed before running)
```bash
python Agent_Caller.py
```

Training
 ===
 
 1. Train baseline model (origin ST Model)
```bash
cd graph_parser
python run_systems_datasets.py --few_shot 5 --dataset cdcp --gpu_id X
```
 2. Auto Annotation (Only for DOS)
 (adjust model_path [baseline ST model path]) 
  Note that dataset and fewshot is in the python file
```bash
python run_distill.py --gpu_id X --model model_path
```

 3.trainning
```bash
python run_pretrain_sft_props --gpu_id X --agent imitation_agent[paraphrase_agent] --dataset cdcp --few_shot 5 
```



Citation
==
```bash

```