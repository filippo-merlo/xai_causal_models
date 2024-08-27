from config import * 
from utils import * 
from dataset import * 

import os
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt

from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)
# Preparation
# Initialize the model and tokenizer
def load_model(model_name, bnb_config, model_dir, cache_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True, cache_dir=cache_dir)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config, MODEL_PATH, CACHE_DIR)

responses = []
answers = []

for idx, example in enumerate(DATASET_SHORT[:]):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0]
    answer_start = example["answers"]["answer_start"]

    eval_prompt = context + ' ' + question

    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers questions accurately and concisely. Respond with only the required name."},
        {"role": "user", "content": f"{eval_prompt}"},
    ]
    model_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            model_input, max_new_tokens=256,
            eos_token_id=terminators)[0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        response = response.split('\n')[-1]

    responses.append(remove_punct(response))
    answers.append(answer)


print(exact_match(responses, answers))
print(match(responses, answers))