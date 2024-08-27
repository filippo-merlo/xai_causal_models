from config import * 
from datasets import load_dataset
import torch 
import numpy as np

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", cache_dir=DATASET_PATH, trust_remote_code=True)
dataset_iter = iter(dataset['train'])

# Get the lenght of the context
lenghts = []
for _ in range(10000):
    example = next(dataset_iter)
    position = example['answers']['answer_start'][0]
    context = example["context"]
    lenghts.append(len(context))
# Print the 99th percentile of the lenghts
print(lenghts)
print(np.percentile(lenghts, 5))



