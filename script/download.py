#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download.py
-----------
此腳本用於下載指定的語言模型及其 tokenizer，
並將它們保存到本地指定的目錄中。
"""

import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from huggingface_hub import login

login(token="")

def download_model(model_name: str, model_path: str) -> bool:
    cache_dir = os.path.join(os.getcwd(), model_path)
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Downloading model '{model_name}' to '{cache_dir}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype='float16',
            device_map="auto"
        )
    except Exception as e:
        logging.error(f"Error downloading model '{model_name}': {e}")
        return False

    logging.info(f"Downloading tokenizer for '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False
        )
    except Exception as e:
        logging.error(f"Error downloading tokenizer for '{model_name}': {e}")
        return False

    logging.info(f"Saving model and tokenizer to '{cache_dir}'...")
    try:
        model.save_pretrained(cache_dir)
        tokenizer.save_pretrained(cache_dir)
    except Exception as e:
        logging.error(f"Error saving model/tokenizer for '{model_name}': {e}")
        return False

    logging.info(f"Model and tokenizer for '{model_name}' downloaded and saved successfully.")
    return True

def main():
    models_to_download = [
        #{
        #    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        #    "model_path": "shared/llama3/1B"
        #},
        {
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "model_path": "shared/mistral/7B",
        },
        #{
        #    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        #    "model_path": "shared/llama3/8B"
        #},
        #{
        #    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        #    "model_path": "shared/llama3/3B"
        #},
    ]
    
    for model_info in models_to_download:
        success = download_model(model_info["model_name"], model_info["model_path"])
        if not success:
            logging.error(f"Failed to download model: {model_info['model_name']}")
    logging.info("All specified models have been downloaded successfully.")

if __name__ == "__main__":
    main()
