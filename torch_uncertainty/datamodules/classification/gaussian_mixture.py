from pathlib import Path
from typing import Literal

from torch.utils.data import random_split

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification.toy.gaussian_mixtue import GaussianMixtureDataset


class GaussianMixtureDatamodule(TUDataModule):
    num_channels = 2  # Input is 2D
    input_shape = (2,)  # For linear input layers
    training_task = "classification"
    mean = (0.0, 0.0)
    std = (1.0, 1.0)

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 128,
        n_classes: int = 5,
        n_samples_per_class: int = 100,
        std: float = 0.1,
        radius: float = 1.0,
        val_split: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.n_classes = n_classes
        self.n_samples_per_class = n_samples_per_class
        self.std = std
        self.radius = radius

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        full = GaussianMixtureDataset(
            n_samples_per_class=self.n_samples_per_class,
            std=self.std,
            radius=self.radius,
            n_classes=self.n_classes,
        )
        n_val = int(len(full) * self.val_split)
        n_train = len(full) - n_val
        self.train, self.val = random_split(full, [n_train, n_val])
        self.test = self.val  # for simplicity, reuse val as test

    def _get_train_data(self):
        return self.train.dataset.data.numpy()

    def _get_train_targets(self):
        return self.train.dataset.targets.numpy()
