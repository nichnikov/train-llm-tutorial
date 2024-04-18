"""
Тестирование работы на предобученной сети с весами "как есть"
"""

import os
import re
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
from datasets import (Dataset, 
                      DatasetDict, 
                      load_dataset)


def mask_apply(Query: str, ShortQuery: str):
    return " ".join(["Вопрос:", "".join(["{", Query, "}"]), "КОРОТКИЙ ВОПРОС:", ShortQuery])

def prompt_create(user_history: str, prompt_df: pd.DataFrame) -> dict:
    """
    """
    # initial_prompt = "Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕРАХ.\nПРИМЕРЫ:\n Как рассчитывается компенсация при увольнении в отпуске?"
    # initial_prompt = "Примеры: " + " ".join([mask_apply(str(d["InitialQuery"]), str(d["ShortQuery"])) for d in prompt_df.to_dict(orient="records")])
    
    examples = "ПРИМЕРЫ" + " ".join([mask_apply(str(d["InitialQuery"]), str(d["ShortQuery"])) for d in prompt_df.to_dict(orient="records")])
    query = "ВОПРОС ПОЛЬЗОВАТЕЛЯ:" "".join(["{", user_history, "}"])
    initial_prompt = "Ты квалифицированный бухгалтер. ПОЛЬЗОВАТЕЛЬ задает бухгалтерский ВОПРОС, ты напишешь КОРОТКИЙ ВОПРОС, как в ПРИМЕРАХ.\nПРИМЕРЫ:\n " + str(examples)
    return {"messages": [{"content": initial_prompt, "role": "system"}, {"content": " ".join(["Вопрос:", query, "напиши только КОРОТКИЙ ВОПРОС"]), "role": "user"}]}

# Load our test dataset
test_data_df = pd.read_csv(os.path.join(os.getcwd(), "data", "vacation_queries_more_20_tokens.csv"), sep="\t")
prompt_df = pd.read_csv(os.path.join("data", "vacation_prompt.csv"), sep="\t")


train_datasets_items = []
for user_history in test_data_df["text"].to_list():
    prompt_dict = prompt_create(user_history, prompt_df)
    train_datasets_items.append(prompt_dict["messages"])

test_datasets = DatasetDict({
    "test": Dataset.from_dict({"messages": train_datasets_items}),
    })

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

rand_idx = randint(0, len(test_datasets["test"]))

print("rand_idx:", rand_idx, "len(test_datasets):", len(test_datasets["test"]))
rand_idx = 1

# # set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

# load into pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Test on sample

test_results = []
for rand_idx in range(len(test_datasets["test"])):
    prompt = pipe.tokenizer.apply_chat_template(test_datasets["test"][rand_idx]["messages"], 
                                                tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, 
                eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    Query = test_datasets["test"][rand_idx]['messages'][1]['content']
    print("outputs:", outputs)
    GeneratedQuery = outputs[0]['generated_text'][len(prompt):].strip()
    print("GeneratedQuery:", GeneratedQuery)
    # GeneratedQuery = outputs["choices"][0]["message"]["content"]
    print(Query)
    only_query = re.search(r'\{(.*?)\}', Query).group(1)
    print(f"Query:\n{only_query}")

    test_results.append({"Query": only_query, "GeneratedQuery": GeneratedQuery})
    test_results_df = pd.DataFrame(test_results)
    print(test_results_df)
    test_results_df.to_csv(os.path.join("test_results", "black_box_vacation_query_prompts_openchat35_assistent.csv"), sep="\t", index=False)