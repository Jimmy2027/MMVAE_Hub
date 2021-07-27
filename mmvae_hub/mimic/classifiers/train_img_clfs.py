# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import accuracy as accuracy_metric, auroc
from torchmetrics.functional.classification import average_precision
from torchmetrics.functional.classification.precision_recall import precision, recall
from torchvision import models

from mmvae_hub.mimic.utils import filter_labels

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
BATCH_SIZE = 64
DL_WORKERS = 8
NUM_EPOCHS = 1
LR = 0.5e-3
NUM_GPUS = 1


# Create Dataset
class MimicIMG(Dataset):
    """
    Custom Dataset for loading the uni-modal mimic text data
    """

    def __init__(self, modality: str, split: str, img_size: int, undersample_dataset: bool, transform: bool):
        """
        split: string, either train, eval or test
        """
        # self.str_label = ['Finding']
        self.str_labels = ['Lung Opacity', 'Pleural Effusion', 'Support Devices']

        # dir_dataset = Path('/Users/Hendrik/Documents/master3/leomed_klugh/files_small_128')
        self.img_size = img_size
        # dir_dataset = Path(f'/mnt/data/hendrik/mimic_scratch/files_small_{img_size}')
        dir_dataset = Path(f'~/klugh/files_small_{img_size}').expanduser()
        fn_img = dir_dataset / f'{split}_{modality}.pt'

        self.labels = filter_labels(pd.read_csv(dir_dataset / f'{split}_labels.csv').fillna(0), self.str_labels,
                                    undersample_dataset=undersample_dataset, split='train')

        self.imgs = torch.load(fn_img)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if transform:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                normalize
            ])

    def __getitem__(self, idx):
        label = torch.from_numpy((self.labels.iloc[idx][self.str_labels].values).astype(int)).float()
        index = self.labels.iloc[idx].name

        img_pa = self.imgs[index, :, :]
        img = torch.cat([img_pa.unsqueeze(0) for _ in range(3)])

        return self.transform(img), label

    def __len__(self):
        return len(self.labels)


class LM(pl.LightningModule):

    def __init__(self, str_labels: list):
        super().__init__()

        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        self.str_labels = str_labels

        model.classifier = nn.Sequential(nn.Linear(num_ftrs, len(str_labels)), nn.Sigmoid())

        self.model = model.to(DEVICE)
        self.criterion = nn.BCELoss().to(DEVICE)

        self.results_dict = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, y = batch
        predictions = self(img)
        loss = self.criterion(predictions, y)
        self.log('train_loss', loss)
        return {'predictions': predictions.cpu(), 'targets': y.cpu(), 'loss': loss}

    def training_epoch_end(self, outputs):
        predictions = torch.Tensor()
        targets = torch.Tensor()
        for elem in outputs:
            predictions = torch.cat((predictions, elem['predictions']), dim=0)
            targets = torch.cat((targets, elem['targets']), dim=0)

        metrics = {'train_auroc': {}, 'train_avg_precision': {}, 'train_acc': {}, 'train_precision': {},
                   'train_recall': {}}
        for idx, label in enumerate(self.str_labels):
            preds_label = predictions[:, idx]
            targets_label = targets[:, idx].int()
            preds_thr = (preds_label > 0.5).int()
            metrics['train_avg_precision'][label] = average_precision(preds_label, targets_label).item()
            metrics['train_auroc'][label] = auroc(preds_label, targets_label).item()
            metrics['train_acc'][label] = accuracy_metric(preds_thr, targets_label).item()
            metrics['train_precision'][label] = precision(preds_thr, targets_label).item()
            metrics['train_recall'][label] = recall(preds_thr, targets_label).item()

        for metric, values in metrics.items():
            self.log(f'{metric}', values, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.criterion(predictions, y)
        self.log('val_loss', loss, prog_bar=True)

        return {'predictions': predictions.cpu(), 'targets': y.cpu()}

    def validation_epoch_end(self, outputs):
        predictions = torch.Tensor()
        targets = torch.Tensor()
        for elem in outputs:
            predictions = torch.cat((predictions, elem['predictions']), dim=0)
            targets = torch.cat((targets, elem['targets']), dim=0)

        metrics = {'auroc': {}, 'avg_precision': {}, 'acc': {}, 'precision': {}, 'recall': {}}
        for idx, label in enumerate(self.str_labels):
            preds_label = predictions[:, idx]
            targets_label = targets[:, idx].int()
            preds_thr = (preds_label > 0.5).int()
            metrics['avg_precision'][label] = average_precision(preds_label, targets_label).item()
            metrics['auroc'][label] = auroc(preds_label, targets_label).item()
            metrics['acc'][label] = accuracy_metric(preds_thr, targets_label).item()
            metrics['precision'][label] = precision(preds_thr, targets_label).item()
            metrics['recall'][label] = recall(preds_thr, targets_label).item()

        for metric, values in metrics.items():
            self.log(f'{metric}', values, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR)


def train_clf(modality: str, img_size: int):
    # temp
    # train_ds = MimicIMG(modality=modality, split='train', img_size=img_size, undersample_dataset=False, transform=True)
    train_ds = MimicIMG(modality=modality, split='eval', img_size=img_size, undersample_dataset=False, transform=False)
    eval_ds = MimicIMG(modality=modality, split='eval', img_size=img_size, undersample_dataset=False, transform=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=DL_WORKERS)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=DL_WORKERS, drop_last=False)

    imgs, label = next(iter(eval_loader))
    print(imgs.shape)
    # Train model

    lightning_module = LM(str_labels=train_ds.str_labels)
    trainer = pl.Trainer(gpus=NUM_GPUS, max_epochs=NUM_EPOCHS,
                         gradient_clip_val=0.5,
                         stochastic_weight_avg=True,
                         callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00,
                                                  patience=5, verbose=True, mode='min')]
                         )

    trainer.fit(lightning_module, train_loader, eval_loader)

    save_dir = Path('state_dicts')
    save_dir.mkdir(exist_ok=True)
    torch.save(lightning_module.model.state_dict(), save_dir / f'{modality}_clf_{img_size}.pth')

    # Evaluate

    predictions = torch.Tensor()
    targets = torch.Tensor()
    lightning_module.model.eval()
    lightning_module.model.to(DEVICE)

    with torch.no_grad():
        for batch in eval_loader:
            x, y = batch
            x = x.to(DEVICE)
            output = lightning_module(x)
            targets = torch.cat((targets, y.cpu()))
            predictions = torch.cat((predictions, output.cpu()))

    for idx, label in enumerate(train_ds.str_labels):
        preds_label = predictions[:, idx]
        y_label = targets[:, idx].int()
        auroc_score = auroc(preds_label, y_label)
        av_precision_score = average_precision(preds_label, y_label)
        preds_thr = (preds_label > 0.5).int()
        acc = accuracy_metric(preds_thr, y_label)
        prec = precision(preds_thr, y_label)
        rec = recall(preds_thr, y_label)
        print(f'auroc__{label}', auroc_score)
        print(f'avg_precision__{label}', av_precision_score)
        print(f'acc__{label}', acc)
        print(f'precision__{label}', prec)
        print(f'recall__{label}', rec)
        print(f'pred_pos__{label}', sum(preds_thr).item())
        print(f'true_pos__{label}', sum(y_label).item())


if __name__ == '__main__':
    # train_clf('pa', 256)
    train_clf('pa', 128)
    # train_clf('lat', 128)
    # train_clf('lat', 256)
