from pathlib import Path
import lightning as L
from typing import Union
from torchvision import transforms
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader

class FoodDataModule(L.LightningDataModule):
    def __init__(self,
                 transforms: transforms.Compose,
                 test_transforms: transforms.Compose,
                 data_dir : Union[str,Path]='../data',
                 num_workers:int=0,
                 batch_size:int=8
                ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_transform = transforms
        self.test_transform = transforms
        self.class_names=None


    def prepare_data(self) -> None:
        """
        prepare dataset download images, and prepare dataset.
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        """
        pass
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(root=self.data_dir/'train',
                                                    transform=self.train_transform
                                                    )
            self.val_dataset = datasets.ImageFolder(root=self.data_dir/'val',
                                                    transform=self.test_transform)
            self.class_names = self.train_dataset.classes
        if stage =="validate" or stage is None:
            self.val_dataset = datasets.ImageFolder(root=self.data_dir/'val',
                                                    transform=self.test_transform)
            self.class_names = self.val_dataset.classes

        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(root=self.data_dir/'test',
                                                 transform=self.test_transform)
            self.class_names = self.test_dataset.classes
        
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_dataset, 
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           pin_memory=True)
        
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.test_dataset, 
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(dataset=self.val_dataset, 
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)
