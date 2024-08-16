import os
import sys
import csv
import json

from copy import deepcopy
from typing import Dict, List, Any


sys.path.append('.')


def csv_to_list(csv_path: str) -> List[List[str]]:
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames
        result_dict = {header: [] for header in headers}
        for row in reader:
            for header in headers:
                result_dict[header].append(
                    str(row[header]) if row[header] else '-')
    org_headers = deepcopy(list(result_dict.keys()))
    for idx, header in enumerate(org_headers):
        header = header.strip()
        if not header:
            org_headers[idx] = f"<unnamed>"

    columns = [key.replace('\n', ' ') for key in org_headers]
    # columns = [key.replace('\n', ' ') for key in result_dict]
    values = list(result_dict.values())
    rows = []
    for i in range(min([len(x) for x in values])):
        rows.append([value[i] for value in values])
    return [columns] + rows

def list_to_pure_dict(table: List[List[str]]) -> Dict[str, List]:
    result = {}
    for ci, c in enumerate(table[0]):
        result[c] = [table[ri][ci] for ri in range(1, len(table))]
    return result


if __name__ == "__main__":
    from utils.table import list_to_dict

    # for part in ['train', 'test']:
    for part in ['train']:
        with open(f"./dataset/WikiTQ/raw/{part}.tsv", 'r', encoding='utf-8') as tsv_file:
            tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
            data = [row for row in tsv_reader]

        results = []
        for idx, row in enumerate(data):
            # table = list_to_dict(csv_to_list(os.path.join(
            #     "./dataset/WikiTQ/raw", row['context'])))
            table = csv_to_list(os.path.join(
                "./dataset/WikiTQ/raw", row['context']))
            # table = list_to_pure_dict(csv_to_list(os.path.join(
            #     "./dataset/WikiTQ/raw", row['context'])))
            results.append({
                "id": f"WikiTQ-{part}-{idx}",
                "table": [{"table": table}],
                "utterance": row['utterance'],
                "answer": [row['targetValue']]
            })

        with open(f'./dataset/WikiTQ/{part}.list.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
