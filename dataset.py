from config import * 
from datasets import load_dataset
import torch 

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", split="validation", cache_dir=DATASET_PATH, trust_remote_code=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

for _ in range(100):
    example = next(iter(dataloader))
    print(example)
