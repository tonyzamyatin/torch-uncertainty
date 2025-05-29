from typing import Literal

from torch import nn

from torch_uncertainty.layers.batch_ensemble import BatchLinear
from torch_uncertainty.models.resnet.std import _BasicBlock, _Bottleneck, _ResNet
from torch_uncertainty.models.resnet.utils import get_resnet_num_blocks

__all__ = ["ll_batched_resnet"]


class _LastLayerBatchedResNet(_ResNet):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        rank: int,
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

        self.linear = BatchLinear(
            in_planes * self.linear_multiplier * block.expansion,
            num_classes,
            num_estimators=num_estimators,
            rank=rank,
        )


def ll_batched_resnet(
    in_channels: int,
    num_classes: int,
    arch: int,
    num_estimators: int,
    rank: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0,
    width_multiplier: float = 1.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: type[nn.Module] = nn.BatchNorm2d,
) -> _LastLayerBatchedResNet:
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
    return _LastLayerBatchedResNet(
        block=block,
        num_blocks=get_resnet_num_blocks(arch),
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        rank=rank,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=int(in_planes * width_multiplier),
        normalization_layer=normalization_layer,
    )
