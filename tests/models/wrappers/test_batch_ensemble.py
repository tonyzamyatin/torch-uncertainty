import pytest
import torch
from torch import nn

from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble


# Define a simple model for testing wrapper functionality (disregarding the actual BatchEnsemble architecture)
class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.r_group = nn.Parameter(torch.randn(in_features))
        self.s_group = nn.Parameter(torch.randn(out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return self.fc(x)

    def reset_parameters(self):
        self.r_group.data.uniform_(-0.1, 0.1)
        self.s_group.data.uniform_(-0.1, 0.1)


# Test the BatchEnsemble wrapper
def test_batch_ensemble():
    in_features = 10
    out_features = 5
    num_estimators = 3
    model = SimpleModel(in_features, out_features)
    wrapped_model = BatchEnsemble(model, num_estimators)

    # Test forward pass
    x = torch.randn(2, in_features)  # Batch size of 2
    logits = wrapped_model(x)
    assert logits.shape == (2 * num_estimators, out_features)

    # Test freeze_shared_parameters
    wrapped_model.freeze_shared_parameters()
    for name, param in wrapped_model.model.named_parameters():
        if "r_group" not in name and "s_group" not in name and "bias" not in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad

    # Test reset_rank1_scaling_factors
    wrapped_model.reset_rank1_scaling_factors()
    for module in wrapped_model.model.modules():
        if hasattr(module, "r_group") and hasattr(module, "s_group"):
            assert torch.all(module.r_group.data >= -0.1) and torch.all(module.r_group.data <= 0.1)
            assert torch.all(module.s_group.data >= -0.1) and torch.all(module.s_group.data <= 0.1)


if __name__ == "__main__":
    pytest.main([__file__])
