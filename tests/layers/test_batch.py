import pytest
import torch

from torch_uncertainty.layers.batch_ensemble import BatchConv2d, BatchLinear


@pytest.fixture()
def feat_input() -> torch.Tensor:
    return torch.rand((4, 6))


@pytest.fixture()
def img_input() -> torch.Tensor:
    return torch.rand((4, 3, 16, 16))


class TestBatchLinear:
    """Testing the BatchLinear layer class."""

    def test_linear_1_estimator(self, feat_input: torch.Tensor):
        layer = BatchLinear(6, 2, num_estimators=1)
        print(layer)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_2_estimators(self, feat_input: torch.Tensor):
        num_estimators = 2
        layer = BatchLinear(6, 2, num_estimators=num_estimators)
        out = layer(feat_input.repeat(num_estimators, 1))
        assert out.shape == torch.Size([8, 2])

    def test_linear_1_estimator_no_bias(self, feat_input: torch.Tensor):
        layer = BatchLinear(6, 2, num_estimators=1, bias=False)
        out = layer(feat_input)
        assert out.shape == torch.Size([4, 2])

    def test_linear_2_estimators_rank_2(self, feat_input: torch.Tensor):
        num_estimators = 2
        rank = 2
        layer = BatchLinear(6, 4, num_estimators=num_estimators, rank=rank)
        out = layer(feat_input.repeat(num_estimators, 1))
        assert out.shape == torch.Size([8, 4])
        assert layer.r_group.shape == (num_estimators, rank, 6)
        assert layer.s_group.shape == (num_estimators, rank, 4)


class TestBatchConv2d:
    """Testing the BatchConv2d layer class."""

    def test_conv_1_estimator(self, img_input: torch.Tensor):
        layer = BatchConv2d(3, 2, num_estimators=1, kernel_size=1)
        print(layer)
        out = layer(img_input)
        assert out.shape == torch.Size([4, 2, 16, 16])

    def test_conv_2_estimators(self, img_input: torch.Tensor):
        num_estimators = 2
        layer = BatchConv2d(3, 2, num_estimators=2, kernel_size=1)
        out = layer(img_input.repeat(num_estimators, 1, 1, 1))
        assert out.shape == torch.Size([8, 2, 16, 16])

    def test_conv_2_estimators_no_bias(self, img_input: torch.Tensor):
        layer = BatchConv2d(3, 2, num_estimators=2, kernel_size=3, bias=False)
        out = layer(img_input.repeat(2, 1, 1, 1))
        assert out.shape == (8, 2, 14, 14)

    def test_conv_2_estimators_rank_2(self, img_input: torch.Tensor):
        num_estimators = 2
        rank = 2
        layer = BatchConv2d(3, 2, num_estimators=num_estimators, rank=rank, kernel_size=3)
        out = layer(img_input.repeat(num_estimators, 1, 1, 1))
        assert out.shape == torch.Size([8, 2, 14, 14])
        assert layer.r_group.shape == (num_estimators, rank, 2, 3, 3)
        assert layer.s_group.shape == (num_estimators, rank, 2, 3, 3)
