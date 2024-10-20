import lightning as pl
import timm
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
import torch.optim as optim
import torch

def get_model(model_name, num_classes, pretrained=True):
    """
    Create base model with pretrained weights from ImageNet if specified.
    """
    model = timm.create_model(model_name, pretrained=pretrained)
    model.reset_classifier(num_classes)
    return model

class TimmClassifier(pl.LightningModule):
    def __init__(
        self,
        base_model: str,
        num_classes: int,
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 3,
        factor: float = 0.1,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = get_model(base_model, num_classes, pretrained)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted')
        self.f1_score = F1Score(task='multiclass', num_classes=num_classes, average='weighted')
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def __common_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_fn(out, y)
        return loss, out, y

    def training_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(out, y)
        f1score = self.f1_score(out, y)
        self.log_dict({
            'train/loss': loss,
            'train/acc': accuracy,
            'train/f1': f1score,
        }, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(out, y)
        f1score = self.f1_score(out, y)
        self.log_dict({
            'val/loss': loss,
            'val/acc': accuracy,
            'val/f1': f1score,
        }, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, out, y = self.__common_step(batch, batch_idx)
        accuracy = self.accuracy(out, y)
        f1score = self.f1_score(out, y)
        self.log_dict({
            'test/loss': loss,
            'test/acc': accuracy,
            'test/f1': f1score,
        }, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        preds = torch.argmax(out, dim=1)
        return preds