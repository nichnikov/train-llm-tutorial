"""
Тестирование работы на предобученной сети с весами "как есть"
"""

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
from datasets import (Dataset, 
                      DatasetDict, 
                      load_dataset)


examples = 'ПОЛЬЗОВАТЕЛЬ: Добрый день.Подскажите пожалуста.У меня работник в отпуске по 28.12.2023 г включительно.Пришел написал заявление на увольнение.Каким числом я его должна уволить 28 или 29 декабря КОРОТКИЙ: Как рассчитывается компенсация при увольнении в отпуске? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день.Подскажите,пожалуйста. Сотрудница вышла в отпуск по уход.за ребенком до 1.5 лет, и сразу же вышла на работу на условиях неполного раб.времени с сохранением пособия. Как это отразить в СЗВ-СТАЖ? Код ДЕТИ не проставляем, верно? И нужно ли указать ставку 0,5?	КОРОТКИЙ: Как отразить в СЗВ-СТАЖ выход на неполный рабочий день из отпуска по уходу до 1.5 лет с сохранением пособия? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день! В какой срок и на какую дату нужно подать форму ЕФС-1 подраздел 1.2, до окончания отчетного периода, на сотрудницу которая подала заявление на отпуск по беременности и родам или по уходу за ребенком до 1,5 лет? КОРОТКИЙ: Срок подачи формы ЕФС-1 подраздел 1.2  заявления на отпуск по родам или по уходу за ребенком до 1,5 лет? \
ПОЛЬЗОВАТЕЛЬ:  у нас трудится внешний совместитель на 0,5 ставки. По основному месту работы он является директором в ООО, в котором деятельность пока не ведется, т.е. он находится в отпуске без сохранения зарплаты. В настоящее время он хочет перейти на полную ставку в нашей компании. Возможно ли это? КОРОТКИЙ: Можно ли совмещать должность директора ООО с другой работой на полной ставке? \
здравствуйте! подскажите пожалуйста Нужно  платить страховые взносы на травматизм с зарплаты и компенсации за не использованный отпуск умершему сотруднику. начисление после смерти сотрудника КОРОТКИЙ: Как платить страховые взносы на травматизм с зарплаты и компенсацию за не использованный отпуск? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день, Директор находится в отпуске, приказ на замещение и право подписи не делали, может ли он утверждать авансовые отчеты и подписывать приказ на увольнение сотрудника КОРОТКИЙ: Может ли директор в отпуске подписывать документы? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день. Подскажите, пожалуйста, нужно ли подавать ЕФС-1 с типом "Назначение выплат по ОСС" на сотрудниц, которые уходят в отпуск по уходу от 1,5 до 3 лет, но работающих неполное рабочее время? Спасибо КОРОТКИЙ: Заполнение ЕФС-1 при уходе в отпуск по беременности и родам и уходу до 1.5 лет работающих неполное рабочее время? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день, подскажите, пожалуйста, если у сотрудника в период отпуска случился больничный лист. Сотрудник не хочет "переносить отпуск на "после окончания БЛ" и выходит на работу. Как правильно создать приказ "на отзыв из отпуска" и тогда мы удерживаем отпускные  и за этот период начисляем оплату по БЛ? Или правильно  сделать приказ на "перенос отпуска", но у нас нет точных дат, так как сотрудник возмет эти дни в каких-то других месяцах в течение года? КОРОТКИЙ: Как начислять и учитывать больничный лист в отпуске? \
ПОЛЬЗОВАТЕЛЬ:  добрый день. Работник целый месяц был в отпуске без сохранения з/пл .Надо ли его указывать в табеле учета рабочего времени? Программа его не заполняет.	Как учитывать отпуск без сохранения зарплаты в табеле учета рабочего времени? \
ПОЛЬЗОВАТЕЛЬ:  Добрый день! Отпуск у работника захватывает дни отдыха с 1 по 8 января 2024г., как считаются эти дни для отпуска? КОРОТКИЙ: Как рассчитывать отпуск в праздничные дни?'

def prompt_create(user_history: str) -> dict:
    """
    example:
    {"messages":
    [{
    "content":"Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕР.\nПРИМЕРЫ:\nКак рассчитывается компенсация при увольнении в отпуске?","role":"system"},
    {"content":"Добрый день.Подскажите пожалуста.У меня работник в отпуске по 28.12.2023 г включительно.Пришел написал заявление на увольнение.Каким числом я его должна уволить 28 или 29 декабря?","role":"user"},
    {"content":"Как отразить в СЗВ-СТАЖ выход на неполный рабочий день из отпуска по уходу до 1.5 лет с сохранением пособия?","role":"assistant"}
    ]}
    """

    # initial_prompt = "Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕРАХ.\nПРИМЕРЫ:\n Как рассчитывается компенсация при увольнении в отпуске?"
    initial_prompt = "Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕРАХ.\nПРИМЕРЫ:\n " + str(examples)
    return {"messages": [{"content": initial_prompt, "role": "system"}, {"content": user_history, "role": "user"}]} 

# Load our test dataset
test_data_df = pd.read_csv(os.path.join(os.getcwd(), "data", "vacation_queries_more_20_tokens.csv"), sep="\t")

print(test_data_df)
print(test_data_df.info())

train_datasets_items = []
for user_history in test_data_df["text"].to_list():
    prompt_dict = prompt_create(user_history)
    train_datasets_items.append(prompt_dict["messages"])

test_datasets = DatasetDict({
    "test": Dataset.from_dict({"messages": train_datasets_items}),
    })

print("test datasets:\n", test_datasets)
print(test_datasets["test"])
print(test_datasets["test"][1])
print(test_datasets["test"][1]["messages"][:2])

model_id = "mistralai/Mistral-7B-v0.1"
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
    prompt = pipe.tokenizer.apply_chat_template(test_datasets["test"][rand_idx]["messages"], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, 
                eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    Query = test_datasets["test"][rand_idx]['messages'][1]['content']
    GeneratedQuery = outputs[0]['generated_text'][len(prompt):].strip()

    print(f"Query:\n{Query}")
    # print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
    print(f"Generated Answer:\n{GeneratedQuery}")

    test_results.append({"Query": Query, "GeneratedQuery": GeneratedQuery})
    test_results_df = pd.DataFrame(test_results)
    print(test_results_df)
    test_results_df.to_csv(os.path.join("test_results", "black_box_vacation_query_prompts.csv"), sep="\t", index=False)