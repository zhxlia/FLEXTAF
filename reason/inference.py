import os
import sys
import json
import random
import argparse

from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
from typing import Dict, List, Any, Tuple


sys.path.append('.')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


TASK = {
    "WikiTQ": "TableQA",
    "ToTTo": "Table-to-Text",
    "TabMWP": "TableQA",
    "TabFact": "Fact-Verification",
    "SciTab": "Fact-Verification",
    "HiTab": "TableQA",
    "FeTaQA": "Table-to-Text",
    "LogicNLG": "Table-to-Text",
    "ToTTo": "Table-to-Text",
    "QTSumm": "Table-to-Text",
    "SciGen": "Table-to-Text",
    "AIT-QA": "TableQA",
    "TabMCQ": "TableQA",
    "WikiSQL": "TableQA",
    "ORKG": "TableQA"
}

TRUE_FALSE = {"Supports": "True", "Refutes": "False"}

def shuffle_table(data: List[Dict[str, Any]], args):
    for d in data:
        if isinstance(d["table"][0]["table"][0], dict):
            if args.shuffle_row and "next" not in d["utterance"] and "previous" not in d["utterance"]:
                d["table"][0]["table"] = random.sample(
                    d["table"][0]["table"], len(d["table"][0]["table"]))
            if args.shuffle_column:
                org_column_name = list(d["table"][0]["table"][0].keys())
                ran_column_name = random.sample(
                    deepcopy(org_column_name), len(org_column_name))
                column_map = {c: ran_column_name[ci]
                              for ci, c in enumerate(org_column_name)}
                d["table"][0]["table"] = [{column_map[c]: row[column_map[c]]
                                           for c in row.keys()} for row in d["table"][0]["table"]]
        elif isinstance(d["table"][0]["table"][0], list):
            if args.shuffle_row and "next" not in d["utterance"] and "previous" not in d["utterance"]:
                # print(d["table"][0]["table"])
                d["table"][0]["table"] = [d["table"][0]["table"][0]] + random.sample(
                    d["table"][0]["table"][1:], len(d["table"][0]["table"])-1)
                # print(d["table"][0]["table"])
            if args.shuffle_column:
                org_column_name = d["table"][0]["table"][0]
                ran_column_idx = random.sample(
                    range(len(org_column_name)), len(org_column_name))
                # ran_column_name = random.sample(deepcopy(org_column_name), len(org_column_name))
                column_map = {ran_column_idx[ci]: ci for ci,
                              c in enumerate(org_column_name)}
                org_table = deepcopy(d["table"][0]["table"])
                d["table"][0]["table"] = [[row[column_map[ci]]
                                           for ci, c in enumerate(row)] for row in org_table]


def trans_table_str(org_table: List[Dict[str, Any]], args):
    if "md" in args.dump_path:
        tr_table = list_to_md(org_table)
    elif "htmlns" in args.dump_path:
        tr_table = list_to_html_no_space(org_table)
    elif "html" in args.dump_path:
        tr_table = list_to_html(org_table)
    elif "text" in args.dump_path:
        tr_table = list_to_text(org_table)
    elif "tuple" in args.dump_path:
        tr_table = list_to_tuple(org_table)
    elif "db" in args.dump_path:
        tr_table = str(Table(org_table[0]["table"], 'information', "sql"))
    elif "csv" in args.dump_path:
        tr_table = list_to_csv(org_table)
    elif "tsv" in args.dump_path:
        tr_table = list_to_tsv(org_table)
    elif "pd" in args.dump_path and "demo" not in args.prompt_type: 
        tr_table = list_to_pd(org_table)
    elif "dict" in args.dump_path:
        tr_table = json.dumps(list_to_dict(org_table), ensure_ascii=False, indent=4)
    else:
        tr_table = json.dumps(org_table[0]["table"], ensure_ascii=False, indent=4)
    return tr_table


def trans_table(org_table: List[Dict[str, Any]], args):
    # if "pd" in args.dump_path and "demo" not in args.prompt_type: 
    #     org_table = list_to_pd(org_table)
    if "dict" in args.dump_path:
        org_table = list_to_dict(org_table)
    elif "list" in args.dump_path:
        org_table = org_table[0]["table"]
    return org_table


def prompt_complete(data: List[Dict[str, Any]], args, prompt_template):
    data_packed: List[Dict[str, Any]] = []
    if "dynamic" in args.prompt_type:
        demo_template = prompt_template.split("---\n\n")[-1]
        with open(args.demo_file, "r", encoding="utf8") as f:
            all_demos: List[Dict[str, Any]] = json.load(f)
        docs = [d["utterance"] for d in all_demos]
        queries = [d["utterance"] for d in data]
        retrs = bm25_similarity_multiprocess(docs, queries, args.shot_num)
        for di, d in enumerate(data):
            demos = "\n\n---\n\n".join([demo_template.replace('<table>', trans_table_str(docs[r[0]]["table"], args)).replace(
                '<utterance>', docs[r[0]]["utterance"]).replace('<answer>', docs[r[0]]["prediction"]) for r in retrs])
            prompt = prompt_template.replace('<demos>', demos).replace('<table>', trans_table_str(d['table'], args)).replace(
                '<utterance>', d["utterance"])
            data_packed.append({
                "id": d['id'],
                "table": trans_table(d["table"], args),
                "source": prompt,
                "answer": d["answer"]
            })
    else:
        for d in data:
            table = trans_table_str(d['table'], args)
            prompt = prompt_template.replace('<table>', table).replace(
                '<utterance>', d["utterance"])
            data_packed.append({
                "id": d['id'],
                "table": trans_table(d["table"], args),
                "source": prompt,
                "answer": d["answer"]
            })

    return data_packed

def extract_rationale(rationale: str):
    return rationale.strip("\n").strip(".").split("the answer is: ")[-1]
    # return rationale.strip("\n").split("the answer is: ")[-1]


if __name__ == '__main__':
    from label.summarize import evaluate, METRIC
    from utils.retrieve import bm25_similarity_multiprocess
    from utils.metric_WikiTQ import to_value_list, check_denotation
    from utils.program import fix_answer, parse_answer, extract_code
    from utils.sql import fix_answer_sql, parse_answer_sql, extract_sql
    from utils.generator import generate_with_llm, consistency, consistency_with_error
    from utils.table import list_to_md, list_to_html, list_to_text, list_to_tuple, Table, list_to_csv, list_to_tsv, list_to_pd, list_to_dict, list_to_html_no_space

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name_or_path", type=str, help="llm path")
    parser.add_argument("--config_path", type=str, help="config path")
    parser.add_argument("--prompt_path", type=str, help="prompt path")
    parser.add_argument("--data_file", type=str, help="data path")
    parser.add_argument("--dump_path", type=str, help="dump path")
    parser.add_argument("--prompt_type", type=str, help="prompt type")
    parser.add_argument("--demo_file", type=str, help="demo file")
    parser.add_argument("--shot_num", type=int, default=3, help="shot num")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--shuffle_row", type=bool,
                        default=False, help="shuffle_row")
    parser.add_argument("--shuffle_column", type=bool,
                        default=False, help="shuffle_column")
    parser.add_argument("--save_n", type=bool,
                        default=False, help="save n")
    parser.add_argument("--random_seed", type=int,
                        default=36, help="random seed")
    args = parser.parse_args()
    set_seed(args.random_seed)

    with open(args.data_file, "r", encoding="utf8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    if args.data_size:
        data = random.sample(data, args.data_size)
    for ds, tk in TASK.items():
        if ds.lower() in args.data_file.lower():
            prompt_file = os.path.join(args.prompt_path, f"{tk}.{args.prompt_type}.txt") if args.prompt_type else os.path.join(
                args.prompt_path, f"{tk}.txt")
            task = tk
            break
    for ds, met in METRIC.items():
        if ds.lower() in args.data_file.lower():
            metric = met
            break
    print(task)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = '\n'.join(line.strip('\n') for line in f).strip()

    shuffle_table(data, args)
    # print(data[-1]["table"][0]["table"])
    data_packed = prompt_complete(data, args, prompt_template)
    print(data_packed[-1]['source'])

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(len([x['source'] for x in data_packed]))
    print(len(data))
    assert len([x['source'] for x in data_packed]) == len(data)
    prediction = generate_with_llm(args.llm_name_or_path, [
                                   x['source'] for x in data_packed], config, 'chat')

    metric_total = 0
    for d, p in tqdm(zip(data_packed, prediction), desc="evaluating", total=len(data_packed)):
        record: List[Tuple[str, str, float]] = []
        if task.lower() in ["tableqa", "fact-verification"] and "nl" not in args.dump_path and "db" not in args.prompt_type:
            for pred in p:
                try:
                    # print(pred[0])
                    # print("%"*88)
                    pred_extracted = extract_code(pred[0])
                    pred_fixed = fix_answer(pred_extracted, d['table'], True if "pd" in args.prompt_type else False)
                    pred_fixed_answer = parse_answer(pred_fixed, d['table'])
                    record.append((pred_fixed, pred_fixed_answer, pred[1]))
                except Exception as e:
                    print(f"{e}")
                    continue
        elif "db" in args.prompt_type:
            for pred in p:
                try:
                    pred_extracted = extract_sql(pred[0])
                    # print(pred_extracted)
                    pred_fixed = fix_answer_sql(pred_extracted, d['table'][0]["table"])
                    pred_fixed_answer = parse_answer_sql(pred_fixed, d['table'][0]["table"])
                    record.append((pred_fixed, pred_fixed_answer, pred[1]))
                except Exception as e:
                    print(f"{e}")
                    continue
        else:
            for pred in p:
                try:
                    pred_fixed = pred[0].strip("\n")
                    pred_fixed_answer = extract_rationale(pred[0])
                    record.append((pred_fixed, pred_fixed_answer, pred[1]))
                except Exception as e:
                    print(f"{e}")
                    continue
        if task.lower() == "fact-verification":
            d["answer"] = [TRUE_FALSE.get(a, a) for a in d["answer"]]
            # print(d["answer"])
        if args.save_n:
            metric_single = 0
            d["predictions"] = [{} for r in record]
            for ri, r in enumerate(record):
                d["predictions"][ri]["prediction"] = r[0].split('\n')
                d["predictions"][ri]["answer_pred"] = r[1]
                if "WikiTQ" in args.data_file:
                    d["predictions"][ri]['correct'] = 1.0 if check_denotation(to_value_list(
                        deepcopy([r[1]])), to_value_list(deepcopy(d['answer']))) else 0.0
                else:
                    d["predictions"][ri]['correct'] = evaluate(str(r[1]), [
                        str(a) for a in d['answer']], metric)
                metric_single += d["predictions"][ri]['correct']

            _, d["answer_pred"] = consistency(record)
            if "WikiTQ" in args.data_file:
                d['correct'] = 1.0 if check_denotation(to_value_list(
                    deepcopy([d['answer_pred']])), to_value_list(deepcopy(d['answer']))) else 0.0
            else:
                d['correct'] = evaluate(str(d['answer_pred']), [str(a) for a in d['answer']], "accuracy")
            metric_total += d['correct']
            d['source'] = d['source'].split('---\n\n')[-1].split("\n")
            d.pop('table')
            # d["correct"] = 1 if metric_single > 0 else 0
        else:
            d["prediction"], d["answer_pred"] = consistency(record)
            d["prediction"] = d["prediction"].split('\n')
            if "WikiTQ" in args.data_file or "WikiSQL" in args.data_file:
                d['correct'] = 1.0 if check_denotation(to_value_list(
                    deepcopy([str(d['answer_pred'])])), to_value_list(deepcopy(([str(d) for d in d['answer']])))) else 0.0
            else:
                d['correct'] = evaluate(str(d['answer_pred']), [
                                    str(a) for a in d['answer']], metric)
            metric_total += d['correct']
            d['source'] = d['source'].split('---\n\n')[-1].split("\n")
            d.pop('table')

            if "answer_pred" not in d.keys():
                d["answer_pred"] = ""
            if not d['answer_pred']:
                try:
                    d['prediction'] = p[0][0].split('\n')
                except Exception as e:
                    continue

    print(f"Average {metric} is: {metric_total} / {len(data_packed)}, {metric_total / len(data_packed)}")
    with open(args.dump_path, 'w', encoding='utf-8') as f:
        json.dump(data_packed, f, indent=4, ensure_ascii=False)
    print(
        f"################################################Results saved to {args.dump_path}################################################")
