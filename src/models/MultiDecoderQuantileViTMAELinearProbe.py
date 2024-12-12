import torch
from torch import nn
import lightning as L
from torchmetrics.functional import accuracy

class MultiDecoderQuantileViTMAELinearProbe(L.LightningModule):
    def __init__(self, pretrained_model, num_classes, learning_rate=1.5e-4):
        """
        Args:
            pretrained_model: A LightningModule wrapping the pretrained ViT-MAE model.
            num_classes: Number of classes for classification.
            learning_rate: Learning rate for the linear layer optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # Freeze the entire pretrained model
        for param in pretrained_model.parameters():
            param.requires_grad = False

        # We assume the pretrained_model has a ViT backbone that we can extract features from.
        # Typically, for ViTMAE, `pretrained_model.model.vit` is the encoder.
        self.feature_extractor = pretrained_model.model.vit

        # The hidden size is needed for the linear layer. Usually in ViT this is available in config.
        hidden_size = self.feature_extractor.config.hidden_size

        # Linear probe layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, pixel_values):
        # Extract features from the pretrained model
        # Typically ViT outputs a dictionary with last_hidden_state
        outputs = self.feature_extractor(pixel_values)
        # Extract the [CLS] token embedding (assuming it's at index 0)
        cls_emb = outputs.last_hidden_state[:, 0]
        # Pass through the linear layer
        logits = self.fc(cls_emb)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        batch_size = pixel_values.shape[0]
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        acc_val_top1 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=1)
        acc_val_top5 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=5)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)
        self.log("train_acc_1", acc_val_top1, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)
        self.log("train_acc_5", acc_val_top5, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        batch_size = pixel_values.shape[0]
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        acc_val_top1 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=1)
        acc_val_top5 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=5)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size = batch_size)
        self.log("val_acc_1", acc_val_top1, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)
        self.log("val_acc_5", acc_val_top5, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch
        batch_size = pixel_values.shape[0]
        logits = self(pixel_values)
        loss = self.criterion(logits, labels)
        acc_val_top1 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=1)
        acc_val_top5 = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes, top_k=5)

        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size = batch_size)
        self.log("test_acc_1", acc_val_top1, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)
        self.log("test_acc_5", acc_val_top5, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)

        return loss

    def configure_optimizers(self):
        # Only the linear layer is trainable
        optimizer = torch.optim.AdamW(self.fc.parameters(), lr=self.learning_rate, weight_decay = 0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.00001)
        return [optimizer], [scheduler]