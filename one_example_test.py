import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from random import randint
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline


example = {"messages":[{"content":"You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.\nSCHEMA:\nCREATE TABLE table_name_16 (total_apps VARCHAR, league_apps VARCHAR, total_goals VARCHAR)","role":"system"},{"content":"What are the total apps that have 41 as the League apps, with total goals greater than 1?","role":"user"},{"content":"SELECT total_apps FROM table_name_16 WHERE league_apps = \"41\" AND total_goals > 1","role":"assistant"}]}

peft_model_id = "./code-mistral-7B-text-to-sql"
# peft_model_id = args.output_dir

# Load Model with PEFT adapter
model = AutoPeftModelForCausalLM.from_pretrained(
  peft_model_id,
  device_map="auto",
  torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load our test dataset
eval_dataset = load_dataset("json", data_files=os.path.join("data", "test_dataset.json"), split="train")
rand_idx = randint(0, len(eval_dataset))

print("example[messages]:", example["messages"][:2])

# Test on sample
# prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
prompt = pipe.tokenizer.apply_chat_template(example["messages"][:2], tokenize=False, add_generation_prompt=True)
print(prompt)

outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, 
               eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

print("outputs:\n", outputs)

print(f"Query:\n{example['messages'][1]['content']}")
print(f"Original Answer:\n{example['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")