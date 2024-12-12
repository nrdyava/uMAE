#!/usr/bin/env python3
import os
import torch
from transformers import ViTMAEConfig
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from runner_utils import start_of_a_run
from lightning.pytorch.strategies import DDPStrategy
from src.models.ViTMAE import ViTMAELightning
from src.datamodules.cifar_100 import DataModule as CIFAR100DataModule
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    config = start_of_a_run()
    
    os.environ['TOKENIZERS_PARALLELISM'] = config['TOKENIZERS_PARALLELISM']
    os.environ['CUDA_LAUNCH_BLOCKING'] = config['CUDA_LAUNCH_BLOCKING']
    os.environ['TORCH_USE_CUDA_DS'] = config['TORCH_USE_CUDA_DS']
    torch.set_float32_matmul_precision(config['float32_matmul_precision'])
    torch.autograd.set_detect_anomaly(True)

    # sets seeds for numpy, torch and python.random.
    seed_everything(config['seed'], workers=True)

    # For training the model
    model_config = ViTMAEConfig()
    #quantiles = config['quantiles']
    model = ViTMAELightning(config=model_config, learning_rate=config['optimizer']['lr'])
    
    datamodule = CIFAR100DataModule(data_dir='./data', batch_size=config['dataloader']['batch_size'], image_size=224, num_workers=config['dataloader']['num_workers'])
    
    wandb_logger = WandbLogger(
        name=config['wandb_name'],
        project=config['wandb']['project'],
        save_dir=config['run_dir']
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['run_dir'],
        filename=config['trainer']['model_checkpoint_callback']['filename'],
        monitor=config['trainer']['model_checkpoint_callback']['monitor'],
        save_top_k=config['trainer']['model_checkpoint_callback']['save_top_k'],
        mode=config['trainer']['model_checkpoint_callback']['mode'],
        every_n_epochs=config['trainer']['model_checkpoint_callback']['every_n_epochs'],
        save_last=True
        )
    
    trainer_args = {
        'accelerator': config['trainer']['accelerator'],
        'strategy': DDPStrategy() if (config['trainer']['strategy'] == 'ddp') else config['trainer']['strategy'],
        'devices': config['trainer']['devices'],
        'num_nodes': config['trainer']['num_nodes'],
        'precision': config['trainer']['precision'],
        'fast_dev_run': config['trainer']['fast_dev_run'],
        'max_epochs': config['trainer']['max_epochs'],
        'min_epochs': config['trainer']['min_epochs'],
        'limit_train_batches': config['trainer']['limit_train_batches'],
        'limit_val_batches': config['trainer']['limit_val_batches'],
        'limit_test_batches': config['trainer']['limit_test_batches'],
        'limit_predict_batches': config['trainer']['limit_predict_batches'],
        'check_val_every_n_epoch': config['trainer']['check_val_every_n_epoch'],
        'num_sanity_val_steps': config['trainer']['num_sanity_val_steps'],
        'log_every_n_steps': config['trainer']['log_every_n_steps'],
        'enable_checkpointing': config['trainer']['enable_checkpointing'],
        'enable_progress_bar': config['trainer']['enable_progress_bar'],
        'enable_model_summary': config['trainer']['enable_model_summary'],
        'deterministic': config['trainer']['deterministic'],
        'benchmark': config['trainer']['benchmark'],
        'use_distributed_sampler': config['trainer']['use_distributed_sampler'],
        'profiler': None if config['trainer']['profiler']=='None' else config['trainer']['profiler'],
        'default_root_dir': config['run_dir'],
        'logger': [wandb_logger], 
        'callbacks': [checkpoint_callback]
        }
    
    trainer = Trainer(**trainer_args)
    
    datamodule.setup('fit')
    trainer.fit(
        model = model, 
        train_dataloaders = datamodule.train_dataloader(), 
        val_dataloaders = datamodule.val_dataloader()
        )