from transformers import ViTMAEConfig, ViTMAEForPreTraining, AutoImageProcessor
import torch
from torch import nn
from copy import deepcopy
import lightning as L


class MultiDecoderQuantileViTMAE(ViTMAEForPreTraining):
    def __init__(self, config, quantiles=[0.1, 0.5, 0.9]):
        super().__init__(config)
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Separate decoders for each quantile
        self.decoders = nn.ModuleList([
            deepcopy(self.decoder) for _ in range(self.num_quantiles)
            ])
        
        self.decoder = None

    def forward_loss(self, pixel_values, preds, mask, interpolate_pos_encoding: bool = False):
        """
        Custom loss for quantile regression with separate decoders.
        Args:
            pixel_values: Original pixel values.
            preds: List of predicted outputs from each decoder.
            mask: Binary mask indicating which patches were masked.
        Returns:
            Combined quantile regression loss.
        """
        target = self.patchify(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        quantile_losses = []
        for i, quantile in enumerate(self.quantiles):
            pred = preds[i]
            diff = target - pred  # (batch, num_patches, patch_dim)
            quantile_loss = torch.max(quantile * diff, (quantile - 1) * diff)  # Pinball loss
            quantile_losses.append(quantile_loss.mean(dim=-1))  # Average over patch_dim

        quantile_loss = sum(quantile_losses) / len(self.quantiles)  # Average over quantiles
        loss = (quantile_loss * mask).sum() / mask.sum()  # Apply mask

        return loss

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        noise: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = None,
        interpolate_pos_encoding: bool = False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the encoder
        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # Forward pass through each decoder
        decoder_outputs = [
            decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
            for decoder in self.decoders
        ]

        preds = [decoder_output.logits for decoder_output in decoder_outputs]  # Predictions from each decoder

        # Calculate combined loss
        loss = self.forward_loss(pixel_values, preds, mask, interpolate_pos_encoding=interpolate_pos_encoding)

        if not return_dict:
            return (loss, preds, mask, ids_restore) + outputs[2:]

        return {
            "loss": loss,
            "preds": preds,
            "mask": mask,
            "ids_restore": ids_restore,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
        
        

class MultiDecoderQuantileViTMAELightning(L.LightningModule):
    def __init__(self, config, quantiles=[0.1, 0.5, 0.9], learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultiDecoderQuantileViTMAE(config, quantiles=quantiles)
        self.learning_rate = learning_rate

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)

    def training_step(self, batch, batch_idx):
        pixel_values, _ = batch  # Unpack CIFAR-100 batch (data, target)
        batch_size = pixel_values.shape[0]
        outputs = self.forward(pixel_values)
        loss = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size = batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, _ = batch  # Unpack CIFAR-100 batch (data, target)
        batch_size = pixel_values.shape[0]
        outputs = self.forward(pixel_values)
        val_loss = outputs["loss"]
        self.log("val_loss", val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size = batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        pixel_values, _ = batch  # Unpack CIFAR-100 batch (data, target)
        batch_size = pixel_values.shape[0]
        outputs = self.forward(pixel_values)
        test_loss = outputs["loss"]
        self.log("test_loss", val_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size = batch_size)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.00001)
        return [optimizer], [scheduler]