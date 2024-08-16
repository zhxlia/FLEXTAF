import os
import sys
import json

sys.path.append('.')

from typing import List, Dict, Any

if __name__ == "__main__":
    from utils.table import list_to_plain_md
    dataset = "WikiTQ"
    # dataset = "TabFact"
    model = "Llama3-chat"
    # model = "Deepseek-Coder-chat"
    scale = "8b"

    for split in ["test"]:
        result_path = f"./reason/results/{dataset}/{model}/{scale}" if scale else f"./reason/results/{dataset}/{model}"
        result_files_list = [f"nl/md/{split}.json", f"pl/db/{split}.changed.json", f"pl/pd/{split}.json", f"pl/list/{split}.json", f"pl/dict/{split}.json"]

        datas: Dict[str, List] = {}

        oup_file = os.path.join(result_path, "compare", f"compare.{split}.json")

        result_files_list = [f for f in result_files_list if os.path.exists(os.path.join(result_path, f))]

        with open(f"./dataset/{dataset}/{split}.list.json", "r", encoding="utf-8") as f:
            org_data = json.load(f)
        org_data = {d["id"]: d for d in org_data}

        for fi, file in enumerate(result_files_list):
            with open(os.path.join(result_path, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            if fi == 0:
                for d in data:
                    datas[d["id"]] = [d]
            else:
                for d in data:
                    datas[d["id"]].append(d)

        statistics = {}
        count_sta = {}
        type_only_correct: List[Dict[str, Any]] = []
        datas_read: List[Dict[str, Any]] = []
        for di, ds in datas.items():
            # print(di)
            statistics[di] = [(result_files_list[pri].split("/")[1], pri) for pri in range(
                len(ds)) if "correct" in ds[pri].keys() and int(ds[pri]["correct"]) == 1]
            # print(statistics[di])
            correct_str = ", ".join([s[0] for s in statistics[di]]) if len(
                statistics[di]) > 0 else "None"
            source: List[str] = f"utterance:\n{org_data[di]['utterance']}\ntable:\n{list_to_plain_md(org_data[di]['table'])}"
            # print(source.split("\n"))
            if "predictions" not in ds[0].keys():
                datas_read.append({"id": di, "correct_type": correct_str, "source": source.split("\n"), "answer": ds[0]["answer"], "predictions": [
                            {"type": result_files_list[ti].split("/")[1], "prediction": d["prediction"], "answer_pred": d["answer_pred"]} for ti, d in enumerate(ds) if "prediction" in d.keys() and "answer_pred" in d.keys()]})
            else:
                datas_read.append({"id": di, "correct_type": correct_str, "source": source.split("\n"), "answer": ds[0]["answer"], "predictions": [
                            {"type": result_files_list[ti].split("/")[1], "prediction": d["predictions"]} for ti, d in enumerate(ds) if "prediction" in d.keys() and "answer_pred" in d.keys()]})
            if len(statistics[di]) == 1:
                if "predictions" not in ds[0].keys():
                    type_only_correct.append({"id": di, "correct_type": correct_str, "source": source.split("\n"), "answer": ds[0]["answer"], "prediction": 
                            ds[statistics[di][0][1]]["prediction"]})
                else:
                    for cp in ds[statistics[di][0][1]]["predictions"]:
                        if int(cp["correct"]) == 1:
                            correct_predictions = cp
                    type_only_correct.append({"id": di, "correct_type": correct_str, "source": source.split("\n"), "answer": ds[0]["answer"], "prediction": 
                            correct_predictions["prediction"].split("\n"), "answer_pred": correct_predictions["answer_pred"], "correct": correct_predictions["correct"], "predictions": ds[statistics[di][0][1]]["predictions"]})
            if correct_str in count_sta.keys():
                count_sta[correct_str] += 1
            else:
                count_sta[correct_str] = 1
        count_st = sorted(count_sta.items(), key=lambda x: x[1], reverse=True)
        indexer_type = [result_files_list[pri].split("/")[1] for pri in range(
            len(result_files_list))]
        per_sta = {ider: count_sta[ider]/sum(
            [count_sta[i] for i in count_sta.keys() if ider in i]) for ider in indexer_type if ider in count_sta.keys()}
        
        oup_sta_file = os.path.join(result_path, "compare", f"compare_sta.{split}.json")

        # print(count_st)
        for c in count_st:
            print(c)
        # print(per_sta)
        for p in per_sta.items():
            print(p)

        type_only_correct_file = os.path.join(result_path, "compare", f"type_only_correct.{split}.json")

        with open(oup_file, 'w', encoding='utf-8') as f:
            json.dump(datas_read, f, indent=4, ensure_ascii=False)

        # with open(oup_sta_file, 'w', encoding='utf-8') as f:
        #     json.dump([count_st, per_sta], f, indent=4, ensure_ascii=False)
        
        # with open(type_only_correct_file, 'w', encoding='utf-8') as f:
        #     json.dump(type_only_correct, f, indent=4, ensure_ascii=False)
