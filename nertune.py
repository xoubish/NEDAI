import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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

# Load the dataset from local files without specifying a script
dataset = load_dataset('text', data_files=data_files)
pdataset = dataset.map(process_sample, batched=True, remove_columns=['text'])

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", num_labels=11, id2label=id2label, label2id=label2id,ignore_mismatched_sizes=True)

# Define a data collator to handle token-level tasks (like NER)
from transformers import DataCollatorForTokenClassification
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

wandb.init(project='NEDAI',name='try1')

model.to(device)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb",  # Log to wandb
    logging_steps=200,
    do_train=True,
    do_eval=True,
    output_dir="./results",
)

# Define the Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = tokenized_datap["train"],
    eval_dataset = tokenized_datap["validation"],
    tokenizer = tokenizer,
)

# Train the model
trainer.train()
wandb.finish()

# Save the model
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")