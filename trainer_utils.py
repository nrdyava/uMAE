from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def get_wandb_logger(config):
    return WandbLogger(
        name=config['wandb_name'],
        project=config['wandb']['project'],
        save_dir=config['run_dir']
        )
    
    
def get_checkpoint_callback(config):
    return ModelCheckpoint(
        dirpath=config['run_dir'],
        filename=config['trainer']['model_checkpoint_callback']['filename'],
        monitor=config['trainer']['model_checkpoint_callback']['monitor'],
        save_top_k=config['trainer']['model_checkpoint_callback']['save_top_k'],
        mode=config['trainer']['model_checkpoint_callback']['mode'],
        every_n_epochs=config['trainer']['model_checkpoint_callback']['every_n_epochs'],
        save_last=True
        )