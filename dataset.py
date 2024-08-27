from config import * 
from datasets import load_dataset
import torch 
import numpy as np

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", cache_dir=DATASET_PATH, trust_remote_code=True)
len_dataset = len(dataset['train'])
dataset_iter = iter(dataset['train'])

# the 1 percentile of the lenghts is 1009.98

# Get the lenght of the context
lenghts = []
DATASET_SHORT = []
for _ in range(len_dataset):
    example = next(dataset_iter)
    position = example['answers']['answer_start'][0]
    context = example["context"]
    lenghts.append(len(context))
    # filter below the 1st percentile
    if len(context) < 1010:
        DATASET_SHORT.append(example)

# Print the 1th percentile of the lenghts
percentile_1th = np.percentile(lenghts, 1)
