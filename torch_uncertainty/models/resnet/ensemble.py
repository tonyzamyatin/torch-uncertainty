from collections.abc import Callable
from typing import Literal

import torch.nn as nn
from torch import relu

from torch_uncertainty.models.resnet.std import resnet
from torch_uncertainty.models.wrappers.deep_ensembles import _DeepEnsembles, deep_ensembles

__all__ = ["ensemble_resnet"]


def ensemble_resnet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    arch: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    width_multiplier: float = 1.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    activation_fn: Callable = relu,
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _DeepEnsembles:
    return deep_ensembles(
        models=resnet(
            in_channels=in_channels,
            num_classes=num_classes,
            arch=arch,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            width_multiplier=width_multiplier,
            groups=groups,
            style=style,
            activation_fn=activation_fn,
            normalization_layer=normalization_layer,
        ),
        num_estimators=num_estimators,
        task="classification",
        reset_model_parameters=True,
    )
