"""
convert dataset from csv to datasets
https://stackoverflow.com/questions/69138037/how-to-load-custom-dataset-from-csv-in-huggingfaces

convert dataset from pandas dataframe to datasets
https://stackoverflow.com/questions/72499850/how-to-load-two-pandas-dataframe-into-hugginfaces-dataset-object

"""

import os
import pandas as pd
from datasets import (Dataset, 
                      DatasetDict, 
                      load_dataset)

prompt_df = pd.read_csv(os.path.join(os.getcwd(), "data", "vacation_prompt.csv"), sep="\t")
print(prompt_df)

datasets_train = DatasetDict({
    "train": Dataset.from_pandas(prompt_df),
    })

print(datasets_train)

# save datasets to disk
datasets_train["train"].to_json(os.path.join(os.getcwd(), "data", "vacation_train_dataset.json"), orient="records")
