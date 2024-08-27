from config import * 
from datasets import load_dataset
import torch 
import numpy as np

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", split="train", cache_dir=DATASET_PATH, trust_remote_code=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

lenghts = []
for _ in range(dataset.__len__()):
    example = next(iter(dataloader))
    position = example['answers']['answer_start'][0]
    context = example["context"][0]
    lenghts.append(context)

print(np.percentile(lenghts, 5))


