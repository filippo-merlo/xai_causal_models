from config import * 
from datasets import load_dataset
import torch 

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", split="train", cache_dir=DATASET_PATH, trust_remote_code=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

for _ in range(1):
    example = next(iter(dataloader))
    print(example)
    position = example['answers']['answer_start'][0]
    context = example["context"][0]
    context = context[:position] + "[!!!!!!!!!START!!!!!!!!]" + context[position:]
    #context = context.split()
    #context.insert(position, "[!!!!!!!!!START!!!!!!!!]") 
    #print(context)


