import json
import torch
import random
import argparse
import accelerate

from transformers import set_seed
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


class CustomDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int, mlb: MultiLabelBinarizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlb = mlb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            inputs = self.tokenizer(item['source'], padding='max_length',
                                    truncation=True, max_length=self.max_length, return_tensors="pt")
            labels = self.mlb.transform([item['labels']])[0]
            result = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(labels, dtype=torch.float)
            }
            return result
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return {}


def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions))
    preds = (preds > 0.5).int()
    labels = torch.tensor(p.label_ids)
    return {'accuracy': (preds == labels).float().mean().item()}


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


label_map = {"md": 0, "db": 1, "dict": 2, "list": 3, "pd": 4}


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
                "source": "\n".join(m["source"]),
                "labels": [label_map.get(l, l) for l in d["correct_type"].split(", ")]
            })
    else:
        for d in data:
            processed_data.append({
                "id": d["id"],
                "source": "\n".join(d["source"]),
                "labels": [label_map.get(l, l) for l in d["correct_type"].split(", ")]
            })
    print(processed_data[0])
    return processed_data

def filter_labels(data: List[Dict[str, Any]], label_low: int, label_high: int):
    if "correct_type" in data[0].keys():
        data = [t for t in data if t["correct_type"] != "None"]
    data = [d for d in data if set(d["correct_type"].split(", ")).intersection(set(label_map.keys()))]
    for d in data:
        d["correct_type"] = ", ".join([l for l in d["correct_type"].split(", ") if l in list(label_map.keys())])
    data = [d for d in data if len(d["correct_type"].split(", ")) >= label_low and len(d["correct_type"].split(", ")) <= label_high]
    return data


if __name__ == "__main__":
    model_name_or_path = "./model/Electra/large"
    tokenizer_name_or_path = "./model/Electra/large"
    output_dir = "./classify/ckpt/WikiTQ/Llama3-chat/8b"
    train_data_file = "./reason/results/WikiTQ/Llama3-chat/8bb/compare/compare.train.json"
    test_data_file = "./reason/results/WikiTQ/Llama3-chat/8b/compare/compare.test.json"
    num_train_epochs = 200
    max_length = 512
    batch_size = 128
    learning_rate =1e-5
    marked = True
    label_low = 1
    label_high = 3
    label_num = 5
    assert label_num == len(list(label_map.keys()))


    # Define the data
    with open(train_data_file, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    train_data = filter_labels(train_data, label_low, label_high)
    with open(test_data_file, 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    test_data = filter_labels(test_data, label_low, label_high)

    marked_train_data = []
    marked_test_data = []
    if marked:
        if "WikiTQ" in train_data_file:
            with open("./dataset/WikiTQ/train.marked.json", 'r', encoding='utf-8') as file:
                marked_train_data = json.load(file)
            with open("./dataset/WikiTQ/test.marked.json", 'r', encoding='utf-8') as file:
                marked_test_data = json.load(file)
        elif "TabFact" in train_data_file:
            with open("./dataset/TabFact/train.marked.json", 'r', encoding='utf-8') as file:
                marked_train_data = json.load(file)
            with open("./dataset/TabFact/test.marked.json", 'r', encoding='utf-8') as file:
                marked_test_data = json.load(file)          
    train_data = process_data(train_data, marked, marked_train_data)
    test_data = process_data(test_data, marked, marked_test_data)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=len(list(label_map.keys())))
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # MultiLabelBinarizer to convert labels to a binary format
    mlb = MultiLabelBinarizer(classes=range(len(list(label_map.keys()))))
    mlb.fit([d['labels'] for d in train_data])

    # max_length = max_length
    # Create datasets
    train_dataset = CustomDataset(
        train_data, tokenizer, max_length=max_length, mlb=mlb)
    test_dataset = CustomDataset(
        test_data, tokenizer, max_length=max_length, mlb=mlb)

    # Verify the dataset outputs
    # for i in range(5):
    #     print(train_dataset[i])

    # Calculate total training steps and eval steps per epoch
    # batch_size = batch_size

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # num_train_epochs = num_train_epochs
    logging_steps = len(train_dataloader)
    eval_steps = 1 * logging_steps

    # DeepSpeed configuration
    deepspeed_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None
        }
    }

    # Save DeepSpeed configuration to file
    with open("./config/ds_config.json", "w") as f:
        json.dump(deepspeed_config, f)

    output_dir=output_dir

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        # total number of training epochs
        num_train_epochs=num_train_epochs,
        logging_dir='./logs',            # directory for storing logs
        logging_steps=logging_steps,
        # eval_steps=eval_steps,           # eval every 10 epochs
        # save_steps=eval_steps,
        deepspeed="./config/ds_config.json",  # path to DeepSpeed configuration file
        save_strategy= 'epoch',
        learning_rate=learning_rate,

    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    # Train the model
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=output_dir)

    # Evaluate the model
    trainer.evaluate()
