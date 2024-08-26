from config import * 
import os
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        cache_dir=model_dir
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

model_name = "meta-llama/Llama-2-13b-chat-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config, MODEL_PATH, CACHE_DIR)

eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    output_ids = model.generate(model_input["input_ids"], max_new_tokens=15)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)

# Perturbation-based Attribution
fa = FeatureAblation(model)

llm_attr = LLMAttribution(fa, tokenizer)

inp = TextTokenInput(
    eval_prompt, 
    tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)

target = "playing guitar, hiking, and spending time with his family."

attr_res = llm_attr.attribute(inp, target=target)

print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)

import matplotlib.pyplot as plt
attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr.png"))
# Optionally, close the plot to free up memory
plt.close()