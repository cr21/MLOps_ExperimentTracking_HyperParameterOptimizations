from pathlib import Path
import lightning as L
from typing import Union, Optional, List, Tuple
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path],
        num_workers: int = 0,
        batch_size: int = 8,
        pin_memory: bool = True,
        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
        splits: List[float] = [0.8, 0.1, 0.1],
        name: str = "generic_datamodule",
        image_size: Tuple[int, int] = (224, 224)

    ):
        super().__init__()
        self.name = name
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.splits = splits
        self.image_size = image_size
        
        # Default transformations
        self.default_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.default_test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.train_transform = train_transform or self.default_train_transform
        self.test_transform = test_transform or self.default_test_transform
        self.class_names = None

    def setup(self, stage: Optional[str] = None):
        if (self.data_dir / 'train').exists() and (self.data_dir / 'test').exists():
            # Train and Test folders exist
            train_dataset = datasets.ImageFolder(root=self.data_dir/'train', transform=self.train_transform)
            test_dataset = datasets.ImageFolder(root=self.data_dir/'test', transform=self.test_transform)
            self.class_names = train_dataset.classes
            if (self.data_dir / 'val').exists():
                # Validation folder also exists
                val_dataset = datasets.ImageFolder(root=self.data_dir/'val', transform=self.test_transform)
            else:
                # Create validation set from train set
                train_size = int(self.splits[0] * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        else:
            #print("ESLSEEE")
            # Create splits from a single directory
            full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transform)
            print(full_dataset.classes)
            train_size = int(self.splits[0] * len(full_dataset))
            val_size = int(self.splits[1] * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
            self.class_names = full_dataset.classes
        if stage == "fit" or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            #self.class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else None
        if stage =="validate" or stage is None:
            self.val_dataset = val_dataset
        if stage == "test" or stage is None:
            self.test_dataset = test_dataset
            #self.class_names = test_dataset.classes if hasattr(train_dataset, 'classes') else None
        print(self.class_names)
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

