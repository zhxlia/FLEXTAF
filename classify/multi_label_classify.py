import json
import torch
import random
import argparse

from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


label_map = {"md": 0, "db": 1, "dict": 2, "list": 3, "pd": 4}
label_map_re = {v: k for k, v in label_map.items()}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, mlb: MultiLabelBinarizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlb = mlb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['source'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = self.mlb.transform([item['labels']])[0]
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def process_data(data: List[Dict[str, Any]], marked: bool, marked_data: List[Dict[str, Any]]):
    processed_data = []
    if marked:
        marked_data = {d["id"]: d for d in marked_data}
        marked_data = [marked_data[d["id"]] for d in data]
        assert len(marked_data) == len(data)
        for d, m in zip(data, marked_data):
            assert d["id"] == m["id"]
            processed_data.append({
                "id": d["id"],
                "correct_type": d["correct_type"],
                "source": "\n".join(m["source"]),
                "labels": [label_map.get(l, l) for l in d["correct_type"].split(", ")]
            })
    else:
        for d in data:
            processed_data.append({
                "id": d["id"],
                "correct_type": d["correct_type"],
                "source": "\n".join(d["source"]),
                "labels": [label_map.get(l, l) for l in d["correct_type"].split(", ")]
            })
    print(processed_data[0])
    return processed_data


def ensemble_predictions(predictions: List[List[int]]) -> List[int]:
    # Compute ensemble predictions
    ensemble_preds = []
    for i in range(len(predictions[0])):
        count = sum([pred[i] for pred in predictions])
        if count > len(predictions) / 2:
            ensemble_preds.append(1)
        else:
            ensemble_preds.append(0)
    return ensemble_preds


def extract_top_x(input_string):
    try:
        if input_string.startswith("top_"):
            return int(input_string.split("_")[1])
        else:
            raise ValueError("Input string format is incorrect.")
    except (IndexError, ValueError) as e:
        print(f"Error: {e}")
        return None
    

def keep_highlog_labels(predictions: List[Any], keep_k: int, correct_type: str) -> List[str]:
    correct_type = correct_type.split(", ")
    high_labels = []
    predictions = [int(p) for p in predictions]
    for pred in predictions:
        pred = int(pred)
        if label_map_re[pred] in correct_type:
            high_labels.append(label_map_re[pred])
        if len(high_labels) == keep_k:
            break   
    return high_labels

def filter_labels(data: List[Dict[str, Any]]):
    # if "correct_type" in data[0].keys():
    #     data = [t for t in data if t["correct_type"] != "None"]
    # data = [d for d in data if set(d["correct_type"].split(", ")).intersection(set(label_map.keys()))]
    for d in data:
        d["correct_type"] = ", ".join([l for l in d["correct_type"].split(", ") if l in list(label_map.keys())])
        if not d["correct_type"]:
            d["correct_type"] = "None"
    return data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument("--tokenizer_name_or_path", type=str, help="tokenizer path")
    parser.add_argument("--test_data_file", type=str, default="./inference/results/WikiTQ/Llama3-chat/70b/compare/compare.test.json", help="test data path")
    parser.add_argument("--dump_path", type=str, help="dump path")
    parser.add_argument("--select_method", type=str, default="top_1", help="select method")
    parser.add_argument("--representation_num", type=int, default=5, help="representation num")
    parser.add_argument("--keep_highlog", type=int, default=0, help="remove lowlog")
    parser.add_argument("--output_log", type=bool, default=False, help="output log")
    parser.add_argument("--marked", type=bool, default=False, help="marked")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    args = parser.parse_args()
    set_seed(args.random_seed)

    with open(args.test_data_file, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    if args.data_size:
        test_data = random.sample(test_data, args.data_size)
    org_test_len = deepcopy(len(test_data))
    test_data = filter_labels(test_data)
    assert args.representation_num == len(list(label_map.keys()))

    marked_test_data = []
    if args.marked:
        if "WikiTQ" in args.test_data_file:
            with open("./dataset/WikiTQ/test.marked.json", 'r', encoding='utf-8') as file:
                marked_test_data = json.load(file)
        elif "TabFact" in args.test_data_file:
            with open("./dataset/TabFact/test.marked.json", 'r', encoding='utf-8') as file:
                marked_test_data = json.load(file)
    test_data = process_data(test_data, args.marked, marked_test_data)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # MultiLabelBinarizer to convert labels to a binary format
    # mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4])
    mlb = MultiLabelBinarizer(classes=range(args.representation_num))
    mlb.fit([d['labels'] for d in test_data])

    # Create test dataset and dataloader
    test_dataset = CustomDataset(test_data, tokenizer, max_length=128, mlb=mlb)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    # Put the model in evaluation mode
    model.eval()

    all_predictions = []
    all_log = []
    k = 1
    if args.keep_highlog or args.output_log:
        k = args.representation_num
    elif "top_" in args.select_method:
        k = extract_top_x(args.select_method)

    # Run predictions on test data
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            outputs = model(**inputs)
            logits = outputs.logits
            if args.keep_highlog or "top_" in args.select_method or args.output_log:
                topk_values, topk_indices = torch.topk(logits, k, dim=-1, largest=True, sorted=True)
                preds = topk_indices.cpu().numpy()
                all_log.extend(topk_values.cpu().numpy())
                # print(preds)
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(preds)
    # print(all_log[:5])
    # Apply ensemble method

    # Prepare the output
    output = []
    total_correct = 0
    if args.keep_highlog:
        for i, example in enumerate(test_data):
            output.append({
                'id': example['id'],
                "correct_type": example['correct_type'],
                'source': example['source'].split("\n"),
                'predicted_labels': keep_highlog_labels(all_predictions[i], args.keep_highlog, example['correct_type']),
                # 'correct': 1 if label_map_re[int(all_predictions[i])] in example['correct_type'] else 0
            })
    elif args.output_log:
        for i, example in enumerate(test_data):
            # print(all_predictions[i])
            # print(all_log[i])
            output.append({
                'id': example['id'],
                "correct_type": example['correct_type'],
                'source': example['source'].split("\n"),
                # 'predicted_labels': [label_map_re[int(p[0])] for p in all_predictions[i]],
                'predicted_labels': {label_map_re[int(p)]: float(all_log[i][pi]) for pi, p in enumerate(all_predictions[i])},
                # 'logits': [l for l in all_log[i]],
                'correct': 1 if label_map_re[int(all_predictions[i][0])] in example['correct_type'] else 0
            })
            total_correct += output[-1]['correct']
    elif k == 1:
        for i, example in enumerate(test_data):
            idx  = [int(p) for p in all_predictions[i]][0]
            output.append({
                'id': example['id'],
                "correct_type": example['correct_type'],
                'source': example['source'].split("\n"),
                'predicted_labels': label_map_re[idx],
                'correct': 1 if label_map_re[idx] in example['correct_type'] else 0
            })
            total_correct += output[-1]['correct']
    print(f"model name: {args.model_name_or_path}")
    print(f"correct number: {total_correct}")
    print(f"classification accucarcy: {total_correct/len([t for t in test_data if t['correct_type'] != 'None'])}")
    print(f"e2e table reasoning accucarcy: {total_correct/org_test_len}")

    # Save the results to a JSON file
    with open(args.dump_path, 'w') as f:
        json.dump(output, f, indent=4)
    print('Results saved to', args.dump_path)
