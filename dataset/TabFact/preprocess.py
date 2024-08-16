import sys
import csv
import json

from typing import List, Dict, Any


sys.path.append('.')


def csv_to_list(file_path: str) -> List[List[str]]:
    result = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='#')
        for row in reader:
            result.append(row)
    return result


if __name__ == '__main__':
    from utils.table import list_to_dict

    # for part in ['train', 'test']:
    for part in ['test']:
        with open(f'./dataset/TabFact/raw/{part}.json', 'r', encoding='utf-8') as f:
            data: Dict[str, List[Any]] = json.load(f)
        with open(f'./dataset/TabFact/raw/small_test_id.json', 'r', encoding='utf-8') as f:
            test_id: List[str] = json.load(f)

        id_map = {}
        results = []
        small_results = []
        for k, d in data.items():
            # table = list_to_dict(csv_to_list(
            #     f'./dataset/TabFact/origin/all_csv/{k}'))
            table = csv_to_list(f'./dataset/TabFact/raw/all_csv/{k}')
            for j, (question, answer) in enumerate(zip(d[0], d[1])):
                results.append({
                    "id": f"TabFact-{part}-{len(results) + 1}",
                    "table": [{
                        "caption": d[2],
                        "table": table
                    }],
                    "utterance": question,
                    "answer": ["Supports" if answer else "Refutes"]
                })
                if k not in test_id:
                    continue
                small_results.append({
                    "id": f"TabFact-{part}-{len(small_results) + 1}",
                    "table": [{
                        "caption": d[2],
                        "table": table
                    }],
                    "utterance": question,
                    "answer": ["Supports" if answer else "Refutes"]
                })
                id_map[f"TabFact-{part}-{len(results)}"] = f"TabFact-{part}-{len(small_results)}"
        with open(f'./dataset/TabFact/{part}.list.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
