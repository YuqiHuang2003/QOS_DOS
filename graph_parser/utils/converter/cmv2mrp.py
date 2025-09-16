# Copyright 2022 by Hitachi, Ltd.
# All rights reserved.

from dataclasses import dataclass, field
from transformers import HfArgumentParser
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Generator
import amparse.common.util as util


@dataclass
class Arguments:
    """
    Arguments
    """
    dir_cmv: str = field(
        default=None,
        metadata={'help': 'The input directory path which contains .txt and .ann files'},
    )
    prefix: str = field(
        default='cmv_',
        metadata={'help': 'The prefix for component labels and edges'},
    )
    output: str = field(
        default=None,
        metadata={'help': 'The output mrp file path'},
    )


def read_brat(ann_path: str, txt_path: str, framework: str, source: str = 'N/A') -> Dict:
    """
    Read the brat formatted file and text file, converting them into the mrp dictionary

    Parameters
    ----------
    ann_path : str
        The path for the brat annotation file (.ann)
    txt_path : str
        The path for the text file associated with the ann_path
    framework : str
        The name of the framework
    prefix : str
        The name of the label prefix
    source : str
        The source of the dataset, e.g., URL

    Returns
    ----------
    mrp : Dict
        The converted mrp dictionary
    """
    with open(ann_path, 'r') as f:
        ann_lines = f.readlines()
    with open(txt_path, 'r') as f:
        txt = f.read()

    nodes, edges, tops = [], [], []
    major_claims = []
    for ann_line in ann_lines:

        annots = ann_line.split('\t')

        if len(annots) < 2:
            assert False, 'Invalid ann format at {}'.format(ann_path)
        if annots[1].startswith('AnnotatorNotes'):
            continue

        # Add component
        if annots[0].startswith('T'):
            adu_data = annots[1].split(' ')

            if len(adu_data) == 3:
                adu_type, start, stop = adu_data
            else:
                adu_type = adu_data[0]
                start, stop = adu_data[1], adu_data[-1]

            if adu_data == 'main_claim':
                adu_type = "claim"

            node = {
                "id": int(annots[0][1:]),
                "label": adu_type,
                "anchors": [{"from": int(start), "to": int(stop)}],
            }
            nodes.append(node)


        # Add relation
        elif annots[0].startswith('R'):
            edge_label, src, trg = annots[1].split(' ')
            src = int(src.replace('Arg1:', '')[1:])
            trg = int(trg.replace('Arg2:', '')[1:])

            if 'agreement' in edge_label:
                edge_label = "support"
            elif "disagreement" in edge_label:
                edge_label = "attack"
            elif "rebuttal" in edge_label:
                edge_label = "attack"
            elif "undercutter" in edge_label:
                edge_label = "attack"

            find = [e for e in edges if e['source'] == src and e['target'] == trg]
            if find:
                logging.warning(f'Found duplication: {ann_path}, {find}')
            else:
                edges.append({"source": src, "target": trg, "label": edge_label})


    # Reassign node id to make the id starts with zero
    nodes = sorted(nodes, key=lambda x: x['anchors'][0]['from'])
    nid2newid = {n['id']: i for i, n in enumerate(nodes)}
    for node in nodes:
        node['id'] = nid2newid[node['id']]
        node['label'] = node['label']
    for edge in edges:
        edge['source'] = nid2newid[edge['source']]
        edge['target'] = nid2newid[edge['target']]
        edge['label'] = edge['label']

    tops = []
    for node in nodes:
        out_edges = [e for e in edges if e['source'] == node['id']]
        if not out_edges:
            tops.append(node['id'])

    mrp = {
        "id": os.path.basename(ann_path).replace('.ann', ''),
        "input": txt,
        "framework": framework,
        "time": "2024-04-04",
        "flavor": 0,
        "version": 1.0,
        "language": "en",
        "provenance": source,
        "source": source,
        "nodes": nodes,
        "edges": edges,
        "tops": tops,
    }
    return mrp


def read_cmv(conf, ann_path: str, txt_path: str):
    mrp = read_brat(ann_path=ann_path, txt_path=txt_path,
                    framework='cmv', source='https://github.com/chridey/change-my-view-modes')

    mrp = util.reverse_edge(mrp=mrp)
    mrp = util.sort_mrp_elements(mrp=mrp)
    return mrp


def main(conf: Arguments):
    # Setup logger
    util.setup_logger(log_dir=None, name=None)
    logging.info(conf)
    # Load files
    ann_files = glob.glob(os.path.join(conf.dir_cmv, '*.ann'), recursive=True)
    txt_files = glob.glob(os.path.join(conf.dir_cmv, '*.txt'), recursive=True)
    # Sort the files
    ann_files, txt_files = sorted(ann_files), sorted(txt_files)
    assert len(ann_files) == len(txt_files)
    logging.info(ann_files)
    logging.info(txt_files)

    mrps, section_mrps = [], []
    for ann, txt in zip(ann_files, txt_files):
        mrp = read_cmv(conf=conf, ann_path=ann, txt_path=txt)
        mrps.append(mrp)

    os.makedirs(os.path.dirname(conf.output), exist_ok=True)
    util.dump_jsonl(fpath=conf.output, jsonl=mrps)
    return


if __name__ == '__main__':
    parser = HfArgumentParser(Arguments)
    conf = parser.parse_args_into_dataclasses()[0]
    main(conf)
