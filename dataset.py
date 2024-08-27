from config import * 
from datasets import load_dataset
import torch 
import numpy as np

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", split="train", cache_dir=DATASET_PATH, trust_remote_code=True)
dataset_iter = iter(dataset)

# Get the lenght of the context
lenghts = []
for _ in range(dataset.__len__()):
    example = next(dataset_iter)
    position = example['answers']['answer_start'][0]
    context = example["context"][0]
    lenghts.append(len(context))
# Print the 99th percentile of the lenghts
print(np.percentile(lenghts, 95))



