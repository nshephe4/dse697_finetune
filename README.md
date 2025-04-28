# Policy Model Fine-Tuning and Evaluation
## By Nathaniel Shepherd
## For DSE 697 GenAI
### Code referenced: https://github.com/0traced/frontier-finetuning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Model-Llama3.8B--Instruct-orange)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-Policy%20Alignment%20Verification-brightgreen)](https://huggingface.co/datasets/nace-ai/policy-alignment-verification-dataset)

---

## Project Overview

This project fine-tunes and evaluates the **Meta Llama 3 8B-Instruct** model for policy and procedure verification tasks.

The repository contains scripts for model fine-tuning, model testing, and a CSV file with comparison results between the base and fine-tuned models.

Results were saved as csv as that allowed for easier viewing than .txt

Note: Will have to download the model and dataset locally before running on odo.

> _"Whatever you do, work heartily, as for the Lord and not for men." — Colossians 3:23_

---

## Contents

| File | Purpose |
|:-----|:--------|
| `finetune.py` | Fine-tunes the base Llama 3 model on the policy alignment dataset. |
| `dataset_load.py` | Downloads the dataset and saves to local cache|
| `model_load.py` | Downloads the model and saves to local|
|`test-subset-sample-data.csv` | 5 samples of the training set|
| `requirements.txt` | Lists the Python dependencies needed to run the scripts. |
| `model_comparison_results.csv` | Stores evaluation results comparing the base model and fine-tuned model. |
| `results_sample.csv` | Stores evaluation results comparing the base model and fine-tuned model (top10 for viewing). |
| `model_test.py` | Runs evaluation tests on the models and outputs results. |

---

## Model Information

- **Base Model:** [Meta Llama 3 8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **Training Dataset:** [Policy Alignment Verification Dataset](https://huggingface.co/datasets/nace-ai/policy-alignment-verification-dataset)

---

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/nshephe4/dse697_finetune.git
   cd yourrepo

2. Download packages

   ```bash
   pip install -r requirements.txt
3. Download dataset
   ```bash
   python dataset_load.py
5. Setup odo allocation
   ```bash
   salloc -A TRN040 -J SFT -t 1:00:00 -p batch -N 1
6. Fine tune model
   ```bash
   python finetune.py

7. Test and eval models
   ```bash
   python model_test.py
