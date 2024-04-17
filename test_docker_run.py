import os
import requests as r
from transformers import AutoTokenizer
from datasets import load_dataset
from random import randint

# Load our test dataset and Tokenizer again
tokenizer = AutoTokenizer.from_pretrained("./code-llama-7b-text-to-sql")
# eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
eval_dataset = load_dataset("json", data_files=os.path.join("data", "test_dataset.json"), split="train")
print("eval_dataset:", eval_dataset)
rand_idx = randint(0, len(eval_dataset))

# generate the same prompt as for the first local test
prompt = tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
print("prompt:", prompt)
request= {"inputs":prompt,"parameters":{"temperature":0.2, "top_p": 0.95, "max_new_tokens": 256}} #256

# send request to inference server
resp = r.post("http://127.0.0.1:8080/generate", json=request)
# resp = r.post("http://0.0.0.0:8080/generate", json=request)

print("resp\n", resp, "\n", resp.json(), "\n")
output = resp.json()["generated_text"].strip()
time_per_token = resp.headers.get("x-time-per-token")
time_prompt_tokens = resp.headers.get("x-prompt-tokens")

# Print results
print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{output}")
print(f"Latency per token: {time_per_token}ms")
print(f"Latency prompt encoding: {time_prompt_tokens}ms")
