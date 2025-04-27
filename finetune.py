import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Model and tokenizer names
base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
new_model_name = "llama-3.18b-policy"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cuda:0",
    cache_dir="llama3-models"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Load dataset
dataset_name = "nace-ai/policy-alignment-verification-dataset"
training_data = load_dataset(dataset_name, split="test", cache_dir="data_cache")

print(training_data.shape)
print(training_data[11])

# 🛠️ Fix the dataset: combine 'query' and 'response' into a 'text' field
def format_example(example):
    return {
        "text": f"Question:\n{example['query']}\n\nAnswer:\n{example['response']}"
    }

training_data = training_data.map(format_example)

# Training parameters
sft_config = SFTConfig(
    output_dir="./results_modified",
    #tokenizer=tokenizer,  # ✅ PASSED HERE
    dataset_text_field="text",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

# LoRA parameters
peft_config = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with PEFT (LoRA)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# Trainer
fine_tuning = SFTTrainer(
    model=model,
    #tokenizer = tokenizer,
    train_dataset=training_data,
    args=sft_config,
)

# Training
fine_tuning.train()

# Save the fine-tuned model
model.save_pretrained("finetuned_llama")
