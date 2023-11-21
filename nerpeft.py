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
    'train': 'data/filtered_train.txt',
    'validation': 'data/filtered_val.txt',
    'test': 'data/filtered_test.txt'
}

model_checkpoint = "dslim/bert-base-NER"
seqeval = evaluate.load("seqeval")

label_list = [
    "O",
    "B-name",
    "I-name",
    "B-redshift",
    "I-redshift",
    "B-RA",
    "I-RA",
    "B-DEC",
    "I-DEC",
    "B-Type",
    "I-Type",
]

id2label = {
    0: "O",
    1: "B-name",
    2: "I-name",
    3: "B-redshift",
    4: "I-redshift",
    5: "B-RA",
    6: "I-RA",
    7: "B-DEC",
    8: "I-DEC",
    9: "B-Type",
    10: "I-Type",
}
label2id = {"O": 0,
          "B-name": 1,
          "I-name": 2,
          "B-redshift": 3,
          "I-redshift": 4,
          "B-RA": 5,
          "I-RA": 6,
          "B-DEC": 7,
          "I-DEC": 8,
          "B-Type": 9,
          "I-Type": 10,
         }

# Load the dataset from local files without specifying a script
dataset = load_dataset('text', data_files=data_files)
pdataset = dataset.map(process_sample, batched=True, remove_columns=['text'])


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=11, id2label=id2label, label2id=label2id,ignore_mismatched_sizes=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

def recursive_label2id_conversion(label, label2id):
    if isinstance(label, str):
        return label2id[label]
    elif isinstance(label, list):
        return [recursive_label2id_conversion(l, label2id) for l in label]
    else:
        raise ValueError("Unsupported label type")

def tokenize_and_align_labels2(examples, label2id):
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

#tokenized_datasets = pdataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_datap = pdataset.map(tokenize_and_align_labels2, batched=True, fn_kwargs={"label2id": label2id},num_proc=4)
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#lr = 1e-3 # I removed lr from training_args so it uses some default?
batch_size = 8
num_epochs = 60

#automatically checks for GPU
training_args = TrainingArguments(
    output_dir="NER-BERT-lora-token-classification",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
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