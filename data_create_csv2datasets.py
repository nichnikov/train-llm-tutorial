"""
convert dataset from csv to datasets
https://stackoverflow.com/questions/69138037/how-to-load-custom-dataset-from-csv-in-huggingfaces

convert dataset from pandas dataframe to datasets
https://stackoverflow.com/questions/72499850/how-to-load-two-pandas-dataframe-into-hugginfaces-dataset-object
https://stackoverflow.com/questions/39612240/writing-pandas-dataframe-to-json-in-unicode

convert dataset from dicts to datasets
https://stackoverflow.com/questions/74779284/convert-dictionary-to-datasets-arrow-dataset-dataset
"""

import os
import pandas as pd
from datasets import (Dataset, 
                      DatasetDict, 
                      load_dataset)


def datasets_item_create(user_history: str, short_query: str) -> list[dict]:
    """
    example:
    {"messages":
    [{
    "content":"Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕР.\nПРИМЕР:\nКак рассчитывается компенсация при увольнении в отпуске?)","role":"system"},
    {"content":"Добрый день.Подскажите пожалуста.У меня работник в отпуске по 28.12.2023 г включительно.Пришел написал заявление на увольнение.Каким числом я его должна уволить 28 или 29 декабря?","role":"user"},
    {"content":"Как отразить в СЗВ-СТАЖ выход на неполный рабочий день из отпуска по уходу до 1.5 лет с сохранением пособия?","role":"assistant"}
    ]}
    """

    initial_prompt = "Ты квалифицированный бухгалтер. Пользователи задают тебе длинный бухгалтерский вопрос на русском языке, ты выберешь из него самое важное и сгенируешь короткий вопрос для поиска ответа в гугле, как в ПРИМЕР.\nПРИМЕР:\n Как рассчитывается компенсация при увольнении в отпуске?)"

    # return {"messages": [{"content": initial_prompt, "role": "system"}, 
    #                     {"content": user_history, "role": "user"}, {"content": short_query, "role": "assistant"}]}
    return [{"content": initial_prompt, "role": "system"}, {"content": user_history, "role": "user"}, {"content": short_query, "role": "assistant"}]




prompts_df = pd.read_csv(os.path.join(os.getcwd(), "data", "vacation_prompt.csv"), sep="\t")
print(prompts_df)

'''
datasets_train = DatasetDict({
    "train": Dataset.from_pandas(prompt_df),
    })'''

train_datasets_items = []
for uh, q in prompts_df.itertuples(index=False, name=None):
    train_datasets_items.append(datasets_item_create(uh, q))

print(train_datasets_items)

datasets_train = DatasetDict({
    "train": Dataset.from_dict({"messages": train_datasets_items}),
    })

print(datasets_train)

# save datasets to disk
datasets_train["train"].to_json(os.path.join(os.getcwd(), "data", "vacation_train_dataset.json"), orient="records", force_ascii=False)
