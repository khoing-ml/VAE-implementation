import os
from pathlib import Path
from typing import Union, Sequence, Optional, List, Any
from PIL import Image
import torch
from torch import Tensor
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets


def repeat_channels(t: Tensor) -> Tensor:
    """Repeat single-channel MNIST tensors to 3 channels for the VAE.

    Defined at module level so it is picklable by DataLoader worker processes.
    """
    return t.repeat(3, 1, 1)


class CelebAFolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.images = sorted(list(self.root.glob("*.jpg")))

        if len(self.images) == 0:
            raise RuntimeError(
                f"No .jpg images found in {self.root}. "
                f"Check your data_path."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # return label 0 because VAE does not need class labels
        return image, 0


class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (64, 64),
        num_workers: int = 0,
        pin_memory: bool = False,
        val_split: float = 0.1,
        dataset_type: str = "celeba",
        mnist_download: bool = True,
        **kwargs
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.dataset_type = dataset_type
        self.mnist_download = mnist_download

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Support both CelebA-style folder datasets and MNIST dataset
        if self.dataset_type.lower() == "mnist":
            # MNIST images are single-channel; repeat channels to match models expecting 3 channels
            train_transforms = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(repeat_channels),
            ])

            val_transforms = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Lambda(repeat_channels),
            ])

            self.train_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=self.mnist_download,
                transform=train_transforms,
            )

            self.val_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=self.mnist_download,
                transform=val_transforms,
            )
        else:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ])

            val_transforms = transforms.Compose([
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ])

            full_train_dataset = CelebAFolderDataset(
                root=self.data_dir,
                transform=train_transforms
            )

            full_val_dataset = CelebAFolderDataset(
                root=self.data_dir,
                transform=val_transforms
            )

            total_size = len(full_train_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            self.train_dataset, _ = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            _, self.val_dataset = random_split(
                full_val_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )