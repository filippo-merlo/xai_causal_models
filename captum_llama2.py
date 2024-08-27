from config import * 
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

#model_name = "meta-llama/Llama-2-13b-chat-hf" 
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config, MODEL_PATH, CACHE_DIR)

eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"

#model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
messages = [
    {"role": "system", "content": "You are an helpful assistant who answewrs questions in a correct and synthetic way.\n"},
    {"role": "user", "content": f"{eval_prompt}\n"},
]
model_input = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
model.eval()
with torch.no_grad():
    output_ids = model.generate(model_input["input_ids"], max_new_tokens=15)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)

### Perturbation-based Attribution
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

attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr_1.png"))
plt.close()

inp = TextTemplateInput(
    template="{} lives in {}, {} and is a {}. {} personal interests include", 
    values=["Dave", "Palm Coast", "FL", "lawyer", "His"],
)

target = "playing golf, hiking, and cooking."

attr_res = llm_attr.attribute(inp, target=target)

attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr_2.png"))
plt.close()

inp = TextTemplateInput(
    template="{} lives in {}, {} and is a {}. {} personal interests include", 
    values=["Dave", "Palm Coast", "FL", "lawyer", "His"],
    baselines=["Sarah", "Seattle", "WA", "doctor", "Her"],
)

attr_res = llm_attr.attribute(inp, target=target)
attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr_3.png"))
plt.close()

baselines = ProductBaselines(
    {
        ("name", "pronoun"):[("Sarah", "her"), ("John", "His"), ("Martin", "His"), ("Rachel", "Her")],
        ("city", "state"): [("Seattle", "WA"), ("Boston", "MA")],
        "occupation": ["doctor", "engineer", "teacher", "technician", "plumber"], 
    }
)

inp = TextTemplateInput(
    "{name} lives in {city}, {state} and is a {occupation}. {pronoun} personal interests include", 
    values={"name":"Dave", "city": "Palm Coast", "state": "FL", "occupation":"lawyer", "pronoun":"His"}, 
    baselines=baselines,
    mask={"name":0, "city": 1, "state": 1, "occupation": 2, "pronoun": 0},
)

attr_res = llm_attr.attribute(inp, target=target, num_trials=3)
attr_res.plot_token_attr(show=False)
plt.savefig(os.path.join(SAVE_IMAGE_PATH,"pert_based_attr_4.png"))
plt.close()