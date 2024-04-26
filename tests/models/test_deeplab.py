import pytest
import torch

from torch_uncertainty.models.segmentation.deeplab import (
    _DeepLabV3,
    deep_lab_v3_resnet50,
    deep_lab_v3_resnet101,
)


class TestDeeplab:
    """Testing the Deeplab class."""

    def test_main(self):
        deep_lab_v3_resnet101(10, "v3+", 8, False, False)
        model = deep_lab_v3_resnet50(10, "v3", 16, True, False).eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError, match="Unknown backbone:"):
            _DeepLabV3(10, "other", "v3", 16, True, False)
