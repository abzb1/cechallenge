import argparse

import json
import ast
import re

from tqdm import tqdm

import torch

# string preprocess
def preprocess(text):
    
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def _process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }
    return out_doc

# read validation data file
def load_hellaswag_jsonl(path):

    # Load jsonl file
    with open(path) as val_file:
        data_lines = val_file.readlines()

    data_dict_lst = []
    
    # convert str to dict  
    for data_line in data_lines:
        converted_data = ast.literal_eval(data_line)
        data_dict_lst.append(converted_data)

    return data_dict_lst

def process_doc(doc):

    processed_dict_lst = []
    doc_bar = tqdm(doc)
    for data_dict in doc_bar:
        processed_dict_lst.append(_process_doc(data_dict))
    doc_bar.close()
    
    return processed_dict_lst

def concat_doc_ctx_cont(doc):

    data_lst = []
    pbar = tqdm(doc)
    for data_dict in pbar:
        sample_lst = []
        ctx = data_dict["query"]
        for choice in data_dict["choices"]:
            sample_lst.append((ctx,choice))
        data_lst.append(sample_lst)
    pbar.close()
    
    return data_lst

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="dataloader",
        description="load hellaswag dataset with preprocessing"
    )
    parser.add_argument("--path",
                        type=str,
                        default="/home1/ohs/cechallenge/inference_base/dataset/hellaswag_val.jsonl",
                        help="data path")

    args = parser.parse_args()

    print("loading data from jsonl")
    ddict = load_hellaswag_jsonl(args.path)
    print("processing data")
    doc = process_doc(ddict)
    print("concatenating contxet and continuation")
    concat_lst = concat_doc_ctx_cont(doc)
    print("doc", len(concat_lst), len(concat_lst[0]), len(concat_lst[0][0]))