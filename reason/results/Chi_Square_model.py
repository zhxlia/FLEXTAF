import os
import json

import pandas as pd
from scipy.stats import chi2_contingency
from typing import Dict, List, Any, Tuple


if __name__=="__main__":
    data_path = "./inference/results/WikiTQ/"
    data_files = ["Llama3-chat/8b/compare/compare.test.json", "Llama3-chat/70b/compare/compare.test.json", "Deepseek-Coder-chat/6.7b/compare/compare.test.json", "Deepseek-Coder-chat/33b/compare/compare.test.json"]
    label = ["md", "dict", "list", "pd", "db"]
    data_count = {}
    data_count["Labels"] = label

    for file in data_files:
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            data = json.load(f)
        label_count = [0, 0, 0, 0, 0]
        for d in data:
            if d["correct_type"] != "None":
                for t in d["correct_type"].split(", "):
                    label_count[label.index(t)] += 1
        data_count[file] = label_count
    df = pd.DataFrame(data_count)
    distribution_matrix = df[data_files].values.T

    chi2, p, dof, expected = chi2_contingency(distribution_matrix)

    print(f"Chi2 Statistic: {chi2}")
    print(f"P-value: {p}")

    if p < 0.05:
        print("数据集的标签分布有显著差异。")
    else:
        print("数据集的标签分布无显著差异。")