import sys
sys.path.append('code/')
from utils import *

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
import torch
import numpy as np

seqeval = evaluate.load("seqeval")

model_checkpoint = "roberta-large"
lr = 1e-3
batch_size = 16
num_epochs = 10

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
    

data_files = {
    'train': 'train.txt',
    'validation': 'val.txt',
    'test': 'test.txt'
}

# Load the dataset from local files without specifying a script
dataset = load_dataset('text', data_files=data_files)
pdataset = dataset.map(process_sample, batched=True, remove_columns=['text'])

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        
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
    9: "B-Type",
    10: "I-Type",
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
          "B-Type": 9,
          "I-Type": 10,
         }

tokenized_datap = pdataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"label2id": label2id})

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=11, id2label=id2label, label2id=label2id)

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

import torch
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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
    compute_metrics=compute_metrics,
)

trainer.train()
