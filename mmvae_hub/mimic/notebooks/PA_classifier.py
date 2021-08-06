# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from pytorch_lightning.metrics.functional import accuracy, precision
from torchmetrics.functional.classification.precision_recall import precision, recall
from torchvision import models

from mmvae_hub.mimic.utils import filter_labels

# %%

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
BATCH_SIZE = 64
DL_WORKERS = 8
NUM_EPOCHS = 100
LR = 0.5e-3


# %% md

# Create Dataset

# %%

class MimicIMG(Dataset):
    """
    Custom Dataset for loading the uni-modal mimic text data
    """

    def __init__(self, modality: str, split: str):
        """
        split: string, either train, eval or test
        """
        self.str_label = ['Finding']

        # dir_dataset = Path('/Users/Hendrik/Documents/master3/leomed_klugh/files_small_128')
        dir_dataset = Path('/mnt/data/hendrik/mimic_scratch/files_small_256')
        fn_img = dir_dataset / f'{split}_{modality}.pt'

        self.labels = filter_labels(pd.read_csv(dir_dataset / f'{split}_labels.csv').fillna(0), self.str_label, False,
                                    'train')

        self.imgs = torch.load(fn_img)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([normalize])

    def __getitem__(self, idx):
        label = torch.from_numpy((self.labels.iloc[idx][self.str_label].values).astype(int)).float()
        index = self.labels.iloc[idx].name

        img_pa = self.imgs[index, :, :]
        img = torch.cat([img_pa.unsqueeze(0) for _ in range(3)])

        return self.transform(img), label

    def __len__(self):
        return len(self.labels)


def train_clf(modality: str):
    train_ds = MimicIMG(modality=modality, split='train')
    eval_ds = MimicIMG(modality=modality, split='eval')

    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

    model = model.to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=DL_WORKERS)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=DL_WORKERS)

    imgs, label = next(iter(eval_loader))
    print(imgs.shape)

    class LM(pl.LightningModule):

        def __init__(self, model):
            super().__init__()
            self.model = model
            self.criterion = nn.BCELoss().to(DEVICE)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            img, y = batch
            logits = self(img)
            loss = self.criterion(logits, y)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1).int()
            acc = accuracy(preds, y.int())
            prec = precision(preds, y.int())
            rec = recall(preds, y.int())

            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            self.log('val_precision', prec, prog_bar=True)
            self.log('val_recall', rec, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            # Here we just reuse the validation_step for testing
            return self.validation_step(batch, batch_idx)

        def configure_optimizers(self):
            return torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Train model

    lightning_module = LM(model=model)
    trainer = pl.Trainer(gpus=1, max_epochs=NUM_EPOCHS, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00,
                                                                                 patience=5, verbose=True, mode='min'),
                                                                   ModelCheckpoint(monitor='val_loss', mode='min',
                                                                                   save_top_k=1)])

    trainer.fit(lightning_module, train_loader, eval_loader)

    torch.save(lightning_module.model.state_dict(), f'{modality}_clf.pth')

    # Evaluate

    predictions, targets = [], []
    model.eval()

    with torch.no_grad():
        for batch in eval_loader:
            x, y = batch
            x = x.to(DEVICE)
            logits = lightning_module(x)
            # take the argmax of the logits
            predictions.extend(logits.argmax(dim=1).tolist())
            targets.extend(y.cpu())

    from sklearn import metrics

    accuracy = metrics.accuracy_score(targets, predictions)
    print("accuracy", accuracy)
    classification_report = metrics.classification_report(targets, predictions)
    print(classification_report)


if __name__ == '__main__':
    train_clf('lat')
