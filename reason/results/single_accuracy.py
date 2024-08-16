import os
import json

import pandas as pd
from scipy.stats import chi2_contingency
from typing import Dict, List, Any, Tuple


if __name__=="__main__":
    file_dirs = ["./inference/results/TabFact/Llama3-chat/8b", "./inference/results/TabFact/Llama3-chat/70b", "./inference/results/TabFact/Deepseek-Coder-chat/6.7b", "./inference/results/TabFact/Deepseek-Coder-chat/33b"]
    for file_dir in file_dirs:
        for rep in ["md", "list", "dict", "pd", "db"]:
            file_path = os.path.join(file_dir, rep, "few", "test.json")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            correct = len([d for d in data if int(d["correct"]) == 1])
            print(f"{file_path}: {correct} / {len(data)}: {correct/len(data)}")
