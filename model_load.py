import os
from huggingface_hub import snapshot_download

# Hugging Face model repo name
model_repo = "meta-llama/Llama-3.1-8B-Instruct"

# Where you want to save the model locally
local_dir = "./local_llama3_8b_instruct"

# Make sure you are logged into Hugging Face first!
# Run 'huggingface-cli login' in your terminal before using this script

print(f"Starting download of {model_repo} into {local_dir}...")

# Download the model
snapshot_download(
    repo_id=model_repo,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # copy full files instead of symlinks (better for moving to clusters)
    resume_download=True,          # in case it fails halfway, can resume
    ignore_patterns=["*.msgpack", "*.safetensors.index.json"]  # optional: skip huge files you don't need
)

print(f"✅ Download completed! Model is saved at: {os.path.abspath(local_dir)}")
