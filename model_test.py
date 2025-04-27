import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Paths
base_model_path = "./local_llama3_8b_instruct"          # path to local base model
finetuned_model_path = "./llama-3.18b-policy"               # path to your fine-tuned model

# Load models
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda:0")
    model.eval()
    return tokenizer, model

print("Loading base model...")
base_tokenizer, base_model = load_model_and_tokenizer(base_model_path)

print("Loading fine-tuned model...")
ft_tokenizer, ft_model = load_model_and_tokenizer(finetuned_model_path)

# Load evaluation dataset
dataset = load_dataset("nace-ai/policy-alignment-verification-dataset", split="test")

# Prepare evaluation prompts
def create_prompt(example):
    return f"""Context:\n{example['context']}\n\nQuestion:\n{example['query']}\n\nAnswer:"""

dataset = dataset.map(lambda x: {"prompt": create_prompt(x)})

# Generation parameters
gen_kwargs = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": False,   # Greedy decoding for fairness
}

# Helper to generate response
def generate_response(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the newly generated text after the prompt
    generated_text = decoded[len(prompt):].strip()
    return generated_text

# Evaluation
base_outputs = []
ft_outputs = []

print("Evaluating models...")
for example in tqdm(dataset):
    prompt = example["prompt"]

    # Base model generation
    base_output = generate_response(base_tokenizer, base_model, prompt)
    base_outputs.append(base_output)

    # Fine-tuned model generation
    ft_output = generate_response(ft_tokenizer, ft_model, prompt)
    ft_outputs.append(ft_output)

# Save outputs for review
import pandas as pd

results_df = pd.DataFrame({
    "query": dataset["query"],
    "context": dataset["context"],
    "gold_response": dataset["response"],
    "base_model_response": base_outputs,
    "finetuned_model_response": ft_outputs,
    "label": dataset["label"],  # optional: if you want to filter compliant vs noncompliant
})

results_df.to_csv("model_comparison_results.csv", index=False)

print("✅ Results saved to model_comparison_results.csv")
