from typing import Literal

import torch
import torchvision.models as tv_models
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights

from torch_uncertainty.models.utils import Backbone


class SeparableConv2d(nn.Module):
    """Separable Convolution with dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias=True,
    ) -> None:
        super().__init__()
        self.separable = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.separable(x))


class InnerConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class InnerPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        x = F.relu(self.bn(self.conv(self.pool(x))))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        atrous_rates: list[int],
        dropout_rate: float,
    ) -> None:
        """Atrous Spatial Pyramid Pooling."""
        super().__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        modules += [
            InnerConv(in_channels, out_channels, rate) for rate in atrous_rates
        ]
        modules.append(InnerPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.projection = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.projection(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = SeparableConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias,
        )
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


class DeepLabV3Backbone(Backbone):
    def __init__(self, backbone_name: str, style: str) -> None:
        # TODO: handle dilations
        if backbone_name == "resnet50":
            base_model = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            base_model = tv_models.resnet101(weights=ResNet101_Weights.DEFAULT)
        base_model.avgpool = nn.Identity()
        base_model.fc = nn.Identity()
        feat_names = ["layer1", "layer4"] if style == "v3+" else ["layer4"]
        super().__init__(base_model, feat_names)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        low_level_channels: int,
        num_classes: int,
        aspp_dilate: list[int] | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        if aspp_dilate is None:
            aspp_dilate = [12, 24, 36]
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.atrous_spatial_pyramid_pool = ASPP(
            in_channels, aspp_dilate, dropout_rate
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        low_level_features = self.project(features[0])
        output_features = self.atrous_spatial_pyramid_pool(features[1])
        output_features = F.interpolate(
            output_features,
            size=low_level_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(
            torch.cat([low_level_features, output_features], dim=1)
        )


class DeepLabV3Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        aspp_dilate: list[int] | None = None,
    ) -> None:
        if aspp_dilate is None:
            aspp_dilate = [12, 24, 36]
        super().__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate, dropout_rate=0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        return self.classifier(features[1])


class DeepLabV3(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        style=Literal["v3", "v3+"],
        output_stride: int = 16,
    ) -> None:
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise ValueError(
                f"output_stride: {output_stride} is not supported."
            )

        self.backbone = DeepLabV3Backbone(backbone_name, style)
        if style == "v3+":
            self.decoder = DeepLabV3PlusDecoder(
                in_channels=2048,
                low_level_channels=256,
                num_classes=21,
                aspp_dilate=dilations,
            )
        elif style == "v3":
            self.decoder = DeepLabV3Decoder(
                in_channels=2048, num_classes=21, aspp_dilate=dilations
            )
        else:
            raise ValueError(f"Unknown style: {style}.")

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        return F.interpolate(
            self.decoder(self.backbone(x)),
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )
