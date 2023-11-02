import sys
sys.path.append('code/')

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

import torch
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
 
def compute_metrics2(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert numerical predictions and labels to their string representations
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Use seqeval to compute metrics
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    # Extract metrics specifically for ap_name1
    ap_name1_metrics = results.get("ap_name1", {"precision": 0, "recall": 0, "f1": 0})

    return {
        "ap_name1_precision": ap_name1_metrics["precision"],
        "ap_name1_recall": ap_name1_metrics["recall"],
        "ap_name1_f1": ap_name1_metrics["f1"],
        "overall_accuracy": results["overall_accuracy"],
    }

def recursive_label2id_conversion(label, label2id):
    if isinstance(label, str):
        return label2id[label]
    elif isinstance(label, list):
        return [recursive_label2id_conversion(l, label2id) for l in label]
    else:
        raise ValueError("Unsupported label type")

def process_sample(batch):
    tokens_list = []
    tags_list = []
    tokens = []
    tags = []
    
    for line in batch['text']:
        if line:  # non-empty line means we have a token-tag pair
            token, tag = line.split()  # assuming space is the delimiter
            tokens.append(token)
            tags.append(tag)
        else:  # empty line means end of sentence
            tokens_list.append(tokens)
            tags_list.append(tags)
            tokens = []
            tags = []
    
    # Add remaining tokens and tags if there's any
    if tokens:
        tokens_list.append(tokens)
        tags_list.append(tags)
    
    return {'tokens': tokens_list, 'tags': tags_list}
    
def tokenize_and_align_labels(examples, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        converted_label = recursive_label2id_conversion(label, label2id)

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(converted_label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

label_list = [
    "O",
    "B-ap_name1",
    "I-ap_name1",
    "B-vz1",
    "I-vz1",
    "B-coordx1",
    "I-coordx1",
    "B-coordy1",
    "I-coordy1",
    "B-type1",
    "I-type1",
]
id2label = {
    0: "O",
    1: "B-ap_name1",
    2: "I-ap_name1",
    3: "B-vz1",
    4: "I-vz1",
    5: "B-coordx1",
    6: "I-coordx1",
    7: "B-coordy1",
    8: "I-coordy1",
    9: "B-type1",
    10: "I-type1",
}

label2id = {"O": 0,
          "B-ap_name1": 1,
          "I-ap_name1": 2,
          "B-vz1": 3,
          "I-vz1": 4,
          "B-coordx1": 5,
          "I-coordx1": 6,
          "B-coordy1": 7,
          "I-coordy1": 8,
          "B-type1": 9,
          "I-type1": 10,
         }

data_files = {
    'train': 'data/train.txt',
    'validation': 'data/val.txt',
    'test': 'data/test.txt'
}

model_checkpoint = "roberta-large"
lr = 1e-3
batch_size = 8
num_epochs = 50


seqeval = evaluate.load("seqeval")

# Load the dataset from local files without specifying a script
dataset = load_dataset('text', data_files=data_files)
pdataset = dataset.map(process_sample, batched=True, remove_columns=['text'])

# Load a tokenizer, data_collator, and map the data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
tokenized_datap = pdataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"label2id": label2id})


model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=11, id2label=id2label, label2id=label2id)
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#automatically checks for GPU
training_args = TrainingArguments(
    output_dir="roberta-large-lora-token-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datap["train"],
    eval_dataset=tokenized_datap["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics2,
)

trainer.train()
