# Policy Model Fine-Tuning and Evaluation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Model-Llama3.8B--Instruct-orange)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![Dataset](https://img.shields.io/badge/Dataset-Policy%20Alignment%20Verification-brightgreen)](https://huggingface.co/datasets/nace-ai/policy-alignment-verification-dataset)

---

## Project Overview

This project fine-tunes and evaluates the **Meta Llama 3 8B-Instruct** model for policy and procedure verification tasks.

The repository contains scripts for model fine-tuning, model testing, and a CSV file with comparison results between the base and fine-tuned models.

Note: Will have to download the model and dataset locally before running on odo.

> _"Whatever you do, work heartily, as for the Lord and not for men." — Colossians 3:23_

---

## Contents

| File | Purpose |
|:-----|:--------|
| `finetune.py` | Fine-tunes the base Llama 3 model on the policy alignment dataset. |
| `requirements.txt` | Lists the Python dependencies needed to run the scripts. |
| `model_comparison_results.csv` | Stores evaluation results comparing the base model and fine-tuned model. |
| `model_test.py` | Runs evaluation tests on the models and outputs results. |

---

## Model Information

- **Base Model:** [Meta Llama 3 8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- **Training Dataset:** [Policy Alignment Verification Dataset](https://huggingface.co/datasets/nace-ai/policy-alignment-verification-dataset)

---

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo

2. Download packages

   ```bash
   pip install -r requirements.txt

3. Fine tune model
   ```bash
   python finetune.py

4. Test and eval models
   ```bash
   python model_test.py
