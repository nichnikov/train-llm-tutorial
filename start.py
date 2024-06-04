import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from random import randint
from trl import setup_chat_format, SFTTrainer
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          TrainingArguments,
                          pipeline)


# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "openchat/openchat_3.5"


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings


# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
