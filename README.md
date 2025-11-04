Exploring Quality and Diversity in Synthetic Data Generation for Argument Mining
======================

EMNLP 2025 Main Conference

Abstract
===

The advancement of Argument Mining (AM) is hindered by a critical bottleneck: the scarcity of structure-annotated datasets, which are expensive to create manually. Inspired by recent successes in synthetic data generation across various NLP tasks, this paper explores methodologies for LLMs to generate synthetic data for AM.

We investigate two complementary synthesis perspectives: a quality-oriented synthesis approach, which employs structure-aware paraphrasing to preserve annotation quality, and a diversity-oriented synthesis approach, which generates novel argumentative texts with diverse topics and argument structures.

Experiments on three datasets show that augmenting original training data with our synthetic data, particularly when combining both quality- and diversity-oriented instances, significantly enhances the performance of existing AM models, both in full-data and low-resource settings.

Moreover, the positive correlation between synthetic data volume and model performance highlights the scalability of our methods.

Project Structure
===
```text
QOS+DOS/
    ├── Data/
    │   ├── aaec_essay/syn_datas
    │   │   ├── imitation_agent/  # 5% and 100% without annotation
    │   │   └── paraphrase_agent/ # 5% and 100%
    │   └── AAEC, CDCP, AbstRCT ...
    │           
    ├── data_mrp        # original datasets from AAEC, CDCP, AbstRCT
    │
    ├── graph_parser    # ST Model
    │
    └── Struct-Syn      # see /Struct-Syn/README.md
```

Quick Start 
===
1. Environment
```bash
pip install -r requirements.txt
```
2. Prepare API keys and base URLs for LLMs in `Struct-Syn/src/LM/config.json`.

3. Use topic utilities to extract original topics, then brainstorm more topics.
```bash
cd Struct-Syn
python Topic_Summary.py
python Topic_Generate.py
```
4. Generate QOS and DOS datasets (adjust agent parameters and workflow as needed before running):
```bash
python Agent_Caller.py
```

Training
===

1. Train baseline model (original ST Model)
```bash
cd graph_parser
python run_systems_datasets.py --few_shot 5 --dataset cdcp --gpu_id X
```
2. Auto-annotation (DOS only)
   - Set `--model` to the baseline ST model path.
   - Dataset and few-shot options are configured in the script.
```bash
python run_distill.py --gpu_id X --model PATH_TO_BASELINE_MODEL
```
3. Training with synthetic data
```bash
python run_pretrain_sft_props --gpu_id X --agent imitation_agent --dataset cdcp --few_shot 5
# or
python run_pretrain_sft_props --gpu_id X --agent paraphrase_agent --dataset cdcp --few_shot 5
```

Citation
===
```bibtex
@inproceedings{bao2025exploring,
  title={Exploring Quality and Diversity in Synthetic Data Generation for Argument Mining},
  author={Bao, Jianzhu and Huang, Yuqi and Sun, Yang and Wang, Wenya and Zhang, Yice and Jin, Bojun and Xu, Ruifeng},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={26592--26615},
  year={2025}
}
```