from typing import Literal

import torch
from torch import nn
from torch.nn import Linear

from torch_uncertainty.models.resnet.std import _BasicBlock, _Bottleneck, _ResNet
from torch_uncertainty.models.resnet.utils import get_resnet_num_blocks

__all__ = ["ll_ensemble_resnet"]


class _LastLayerEnsembleResNet(_ResNet):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        width_multiplier: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        in_planes: int = 64,
        normalization_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            in_channels=in_channels,
            num_classes=num_classes,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            style=style,
            in_planes=int(in_planes * width_multiplier),
            normalization_layer=normalization_layer,
        )

        # num_estimator independent linear layers
        self.num_estimators = num_estimators
        self.num_classes = num_classes
        self.linears = nn.ModuleList(
            [
                Linear(
                    in_planes * self.linear_multiplier * block.expansion,
                    num_classes,
                    bias=conv_bias,
                )
                for _ in range(num_estimators)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats_forward(x)
        return torch.cat([linear(x) for linear in self.linears], dim=0)

    #     # Stack weights and biases for vectorized operations
    #     self.register_buffer(
    #         "stacked_weights",
    #         torch.stack([linear.weight for linear in self.linears], dim=0),
    #     )   # Shape: (num_estimators, num_classes, feature_dim)
    #     if conv_bias:
    #         self.register_buffer(
    #             "stacked_biases",
    #             torch.stack([linear.bias for linear in self.linears], dim=0),
    #         )   # Shape: (num_estimators, num_classes)
    #     else:
    #         self.stacked_biases = None

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.feats_forward(x)  # Extract features

    #     # Perform batched matrix multiplication
    #     x = x.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
    #     out = torch.einsum("bnd,emd->bem", x, self.stacked_weights) # Shape: (batch_size, num_estimators, num_classes)

    #     # Add biases if applicable
    #     if self.stacked_biases is not None:
    #         out += self.stacked_biases

    #     out = out.view(-1, self.num_classes)  # Shape: (batch_size * num_estimators, num_classes)

    #     return out

    def to(self, *args, **kwargs):
        """Ensure buffers are moved to the correct device."""
        super().to(*args, **kwargs)
        self.stacked_weights = self.stacked_weights.to(*args, **kwargs)
        if self.stacked_biases is not None:
            self.stacked_biases = self.stacked_biases.to(*args, **kwargs)


def ll_ensemble_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    width_multiplier: float = 1.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _LastLayerEnsembleResNet:
    """BatchEnsemble of ResNet.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes to predict.
        arch (int): The architecture of the ResNet.
        num_estimators (int): Number of estimators in the ensemble.
        conv_bias (bool): Whether to use bias in convolutions. Defaults to
            ``True``.
        dropout_rate (float): Dropout rate. Defaults to 0.
        width_multiplier (float): Width multiplier. Defaults to 1.
        groups (int): Number of groups within each estimator.
        style (bool, optional): Whether to use the ImageNet
            structure. Defaults to ``True``.
        normalization_layer (nn.Module, optional): Normalization layer.

    Returns:
        _BatchedResNet: A BatchEnsemble-style ResNet.
    """
    block = _BasicBlock if arch in [18, 20, 34, 44, 56, 110, 1202] else _Bottleneck
    in_planes = 16 if arch in [20, 44, 56, 110, 1202] else 64
    return _LastLayerEnsembleResNet(
        block=block,
        num_blocks=get_resnet_num_blocks(arch),
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=int(in_planes * width_multiplier),
        normalization_layer=normalization_layer,
    )
