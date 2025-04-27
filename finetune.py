import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Model and tokenizer names

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
new_model_name = "llama-3.18b-policy" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cuda:0",
    cache_dir="llama3-models"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Data set
data_name = "nace-ai/policy-alignment-verification-dataset"
training_data = load_dataset(data_name, split="test", cache_dir="data_cache")
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])

# Training Params
train_params = SFTConfig(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    #optim="paged_adamw_32bit",
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
    dataset_text_field="text",
)

from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    processing_class=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()
model.save_pretrained("finetuned_llama")