import os
import json
from datasets import load_dataset


ds = load_dataset("json", data_files=os.path.join("data", "test_dataset.json"))
print(ds)