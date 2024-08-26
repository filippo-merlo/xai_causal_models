from config import * 
from datasets import load_dataset

# Load the Quoref dataset
dataset = load_dataset("allenai/quoref", cache_dir=DATASET_PATH)

print(dataset)