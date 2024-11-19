import torch
from transformers import AutoModelForCausalLM
from llava.model import *
# Paths to the models
template_model_path = "/home/niudt/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
target_model_path = "/home/niudt/project/llarva_more/mirage/ckpts/mirage-llama3.1-8.3B"
output_model_path = "/home/niudt/project/llarva_more/mirage/ckpts/mirage-llama3.1-8.3B_main"

# Load the template model
template_model = AutoModelForCausalLM.from_pretrained(template_model_path, low_cpu_mem_usage=True)

# Load the target model
target_model = model = LlavaLlamaForCausalLM.from_pretrained(
                    target_model_path,
                    low_cpu_mem_usage=True,
                )

# Function to update weights
def copy_weights(template_model, target_model):
    # Ensure template and target model have aligned parameters
    template_state_dict = template_model.state_dict()
    target_state_dict = target_model.state_dict()

    for name, param in template_state_dict.items():
        if name in target_state_dict:
            param.data = target_state_dict[name].data.clone()
            print(f"Updated weights for: {name}")
        else:
            print(f"Skipped: {name} (no matching parameter in template model)")

# Copy weights
copy_weights(template_model, target_model)

# Save the updated target model
template_model.save_pretrained(output_model_path)
print(f"Updated target model saved to: {output_model_path}")
