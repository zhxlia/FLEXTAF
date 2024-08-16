import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from matplotlib.colors import LinearSegmentedColormap


if __name__=="__main__":
    data_file = "./inference/results/WikiTQ/Llama3-chat/8b/compare/compare.test.json"
    oup_file = "./inference/results/WikiTQ/Llama3-chat/8b/compare/overlap.label.test.json"

    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = ["MD", "Dict", "List", "PD", "DB"]
    label_overlap = {l: {ll: 0 for ll in labels} for l in labels}
    label_count = {l: 0 for l in labels}
    for l in labels:
        label_count[l] = len([d for d in data if l.lower() in d["correct_type"].split(", ")])
        for ll in labels:
            label_overlap[l][ll] = len([d for d in data if l.lower() in d["correct_type"].split(", ") and ll.lower() in d["correct_type"].split(", ")])
    print(label_overlap)
    print(label_count)
    label_overlap = {k: {kk: v[kk]/label_count[k] for kk in v.keys()} for k, v in label_overlap.items()}
    print(label_overlap)

    data = pd.DataFrame(label_overlap).T
    cmap = LinearSegmentedColormap.from_list("lightblue_to_lightgreen", ["lightblue", "cornflowerblue"])

    # 创建热图
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(data, annot=True, cmap=cmap, fmt=".2f")
    plt.savefig("./inference/results/WikiTQ/Llama3-chat/8b/compare/heatmap_wtq_8b.pdf", format='pdf', bbox_inches='tight')
    plt.close()
