from config import * 
from datasets import load_dataset
import torch 
import numpy as np

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", cache_dir=DATASET_PATH, trust_remote_code=True)
len_dataset = len(dataset['train'])
dataset_iter = iter(dataset['train'])

# the 1 percentile of the lenghts is 1009.98

# filter examples by lenght
short_context_dataset = []
for _ in range(len_dataset):
    example = next(dataset_iter)
    position = example['answers']['answer_start'][0]
    context = example["context"]
    if len(context) < 1010:
        short_context_dataset.append(example)
        print(context)

print(len(short_context_dataset))


