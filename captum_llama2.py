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

model_name = "meta-llama/Llama-2-13b-chat-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config, MODEL_PATH, CACHE_DIR)

eval_prompt = """Answer Q1 looking at the following text: Byzantines were avid players of tavli (Byzantine Greek:
τάβλη), a game known in English as backgammon, which is still
popular in former Byzantine realms, and still known by the
name tavli in Greece. Byzantine nobles were devoted to
horsemanship, particularly tzykanion, now known as polo.
The game came from Sassanid Persia in the early period and a
Tzykanisterion (stadium for playing the game) was built by
Theodosius II (r. 408–450) inside the Great Palace of
Constantinople. Emperor Basil I (r. 867–886) excelled at it;
Emperor Alexander (r. 912–913) died from exhaustion while
playing, Emperor Alexios I Komnenos (r. 1081–1118) was
injured while playing with Tatikios, and John I of Trebizond (r.
1235–1238) died from a fatal injury during a game. Aside from
Constantinople and Trebizond, other Byzantine cities also
featured tzykanisteria, most notably Sparta, Ephesus, and
Athens, an indication of a thriving urban aristocracy.
Q1. What is the Byzantine name of the game that Emperor
Basil I excelled at?"""

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

target = "it → tzykanion"

attr_res = llm_attr.attribute(inp, target=target)

print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)

import matplotlib.pyplot as plt
attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr.png"))
# Optionally, close the plot to free up memory
plt.close()