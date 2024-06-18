# ruff: noqa: F401
from .baseline import (
    DummyClassificationBaseline,
    DummyPixelRegressionBaseline,
    DummyRegressionBaseline,
    DummySegmentationBaseline,
)
from .datamodule import (
    DummPixelRegressionDataModule,
    DummyClassificationDataModule,
    DummyRegressionDataModule,
    DummySegmentationDataModule,
)
from .dataset import (
    DummPixelRegressionDataset,
    DummyClassificationDataset,
    DummyRegressionDataset,
    DummySegmentationDataset,
)
from .model import dummy_model
from .transform import DummyTransform
