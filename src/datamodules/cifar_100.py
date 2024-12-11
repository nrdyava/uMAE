import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=128, image_size=224, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

        # Define transformations
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def prepare_data(self):
        # Download the data (only called once)
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Called on every GPU separately - good for setting up datasets
        """
        if stage == 'fit' or stage is None:
            self.cifar100_train = datasets.CIFAR100(
                self.data_dir, train=True, transform=self.train_transforms
            )
            self.cifar100_val = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.test_transforms
            )

        if stage == 'test' or stage is None:
            self.cifar100_test = datasets.CIFAR100(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar100_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar100_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )