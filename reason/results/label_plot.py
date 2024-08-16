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


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import List


from matplotlib.backends.backend_pdf import PdfPages

def save_multiple_pie_charts_to_pdf(data_list, figure_names: List[str], filename:str):
    pastel_colors = ["#F0F0F2", "#0583F2", "#0597F2", "#05AFF2", "#F2BC1B"] # blue orange

    num_charts = len(data_list)
    rows = (num_charts + 3) // 4  # Calculate the number of rows needed
    
    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        axes = axes.flatten()  # Flatten to easily iterate through subplots

        for i, data in enumerate(data_list):
            labels = data.keys()
            sizes = data.values()
            colors = pastel_colors[:len(labels)]
            # axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            wedges, texts, autotexts = axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
            for text in texts + autotexts:
                text.set_fontsize(13)
            axes[i].set_title(figure_names[i], y=-0.1, fontsize=13)
            axes[i].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

if __name__=="__main__":
    data_files = ["./inference/results/WikiTQ/Llama3-chat/8b/compare/compare.train.json", "./inference/results/WikiTQ/Llama3-chat/70b/compare/compare.train.json", 
                  "inference/results/WikiTQ/Deepseek-Coder-chat/6.7b/compare/compare.train.json", "./inference/results/WikiTQ/Deepseek-Coder-chat/33b/compare/compare.train.json",
                  "./inference/results/TabFact/Llama3-chat/8b/compare/compare.train.json", "./inference/results/TabFact/Llama3-chat/70b/compare/compare.train.json",
                  "./inference/results/TabFact/Deepseek-Coder-chat/6.7b/compare/compare.train.json", "./inference/results/TabFact/Deepseek-Coder-chat/33b/compare/compare.train.json"]
    figures_names = ["(a) Llama3-8B, WikiTQ", "(b) Llama3-70B, WikiTQ", "(c) Deepseek-Coder-6.7B, WikiTQ", "(d) Deepseek-Coder-33B, WikiTQ",
                     "(e) Llama3-8B, TabFact", "(f) Llama3-70B, TabFact", "(g) Deepseek-Coder-6.7B, TabFact", "(h) Deepseek-Coder-33B, TabFact"]

    proportion_list = []
    for data_file in data_files:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(data_file)
        labels = ["MD", "Dict", "List", "PD", "DB"]
        label_count = {k: 0 for k in labels}
        data = [d for d in data if d["correct_type"] != "None" and len(d["correct_type"].split(", ")) < 4]
        for d in data:
            d["correct_type"] = d["correct_type"].replace("md", "MD").replace("dict", "Dict").replace("list", "List").replace("pd", "PD").replace("db", "DB")
            for l in d["correct_type"].split(", "):
                label_count[l] = label_count[l] + 1
        labels_sum = sum([label_count[l] for l in labels])
        label_count = {k: v/labels_sum for k, v in label_count.items()}
        print(label_count)
        proportion_list.append(label_count)
    # print(label_overlap)
    

    save_multiple_pie_charts_to_pdf(proportion_list, figures_names, "./inference/results/WikiTQ/compare/plots_bo.pdf")
