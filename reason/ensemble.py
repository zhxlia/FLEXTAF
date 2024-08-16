import os
import sys
import json

sys.path.append('.')

from copy import deepcopy
from typing import List, Dict, Any, Tuple

if __name__ == "__main__":
    from utils.generator import consistency
    from utils.metric_WikiTQ import check_denotation, to_value_list
    with open("./reason/results/WikiTQ/Llama3r-chat/8b/compare/compare.test.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    candidate_type = ["md", "db", "dict", "pd", "list"]
    correct = 0
    oracle = 0
    for d in data:
        d["predictions"] = [p for p in d["predictions"] if p["type"] in candidate_type]
        record: List[Tuple[str, Any, float]] = [("\n".join(p["prediction"]), p["answer_pred"], 1.0) for p in d["predictions"]]
        d["prediction"], d["answer_pred"] = consistency(record)
        d["prediction"] = d["prediction"].split('\n')
        d['correct'] = 1.0 if check_denotation(to_value_list(deepcopy([d['answer_pred']])), to_value_list(deepcopy(d['answer']))) else 0.0
        correct += d["correct"]
        if len([t for t in d["correct_type"].split(", ") if t in candidate_type]) > 0:
            oracle += 1
    print(correct)
    print(oracle)
    print(len(data))
    print(f"ensemble accuracy: {correct / len(data)}")
    print(f"oracle accuracy: {oracle / len(data)}")
