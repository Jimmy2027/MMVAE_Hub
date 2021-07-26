from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from mmvae_hub.mimic.utils import filter_labels

# %%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 10
NUM_EPOCHS = 1

# %%

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


# %% md

# Create Dataset

# %%

class MimicFindings(Dataset):
    """
    Custom Dataset for loading the uni-modal mimic text data
    """

    def __init__(self, split: str):
        """
        split: string, either train, eval or test
        """
        str_label = ['Finding']
        # dir_dataset = Path('/Users/Hendrik/Documents/master3/leomed_klugh/files_small_128')
        dir_dataset = Path('/mnt/data/hendrik/mimic_scratch/files_small_128')
        findings = pd.read_csv(dir_dataset / f'{split}_findings.csv')
        labels = filter_labels(pd.read_csv(dir_dataset / f'{split}_labels.csv').fillna(0), str_label, False, 'train')

        self.df = labels.merge(findings)

        # tokenize findings
        self.encodings = tokenizer(self.df['findings'].tolist(), return_tensors="pt", padding=True, truncation=True,
                                   max_length=256)
        self.labels = self.df['Finding'].tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.df)


# %%

train_ds = MimicFindings('train')
eval_ds = MimicFindings('eval')
train_ds.df.head()

# %%

# Load the model from a pretrained checkpoint.

# %%

unique_labels, counts = np.unique(train_ds.df["Finding"], return_counts=True)
print(unique_labels, counts)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_labels)).to(DEVICE)

optimizer = torch.optim.Adam([
    {'params': model.distilbert.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])

# %%

eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

# %%

# Taken from https://huggingface.co/transformers/custom_datasets.html

# %%

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=NUM_EPOCHS,  # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_ds,  # training dataset
    eval_dataset=eval_ds  # evaluation dataset
)

trainer.train()

# %%

trainer.evaluate()

# %%

model = trainer.model

# %% md
# Save model

# %%
torch.save(model.state_dict(), 'text_clf.pth')

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(unique_labels)).to(DEVICE)

model.load_state_dict(torch.load('text_clf.pth', map_location=DEVICE))

# %%

predictions, targets = [], []
model.eval()

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        # print(output)
        logits = output.logits
        # take the argmax of the logits
        predictions.extend(logits.argmax(dim=1).tolist())
        targets.extend(labels.cpu())

from sklearn import metrics

accuracy = metrics.accuracy_score(targets, predictions)
print("accuracy", accuracy)
classification_report = metrics.classification_report(targets, predictions)
print(classification_report)
