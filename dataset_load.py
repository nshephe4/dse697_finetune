from datasets import load_dataset

# First time: download and cache it locally
dataset = load_dataset("nace-ai/policy-alignment-verification-dataset", split="test", cache_dir="./data_cache"