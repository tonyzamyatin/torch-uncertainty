import torch
from torch import nn


class BatchEnsemble(nn.Module):
    """Wraps a BatchEnsemble model to ensure correct batch replication.

    In a BatchEnsemble architecture, each estimator operates on a **sub-batch**
    of the input. This means that the input batch must be **repeated**
    :attr:`num_estimators` times before being processed.

    This wrapper automatically **duplicates the input batch** along the first axis,
    ensuring that each estimator receives the correct data format.

    **Usage Example:**
    ```python
    model = lenet(in_channels=1, num_classes=10)
    wrapped_model = BatchEnsembleWrapper(model, num_estimators=5)
    logits = wrapped_model(x)  # `x` is automatically repeated `num_estimators` times
    ```

    Args:
        model (nn.Module): The BatchEnsemble model.
        num_estimators (int): Number of ensemble members.
    """

    def __init__(self, model: nn.Module, num_estimators: int):
        super().__init__()
        self.model = model
        self.num_estimators = num_estimators

    def freeze_shared_parameters(self):
        """Freezes the shared parameters of the model (also called slow weights)."""
        for name, param in self.model.named_parameters():
            if "r_group" not in name and "s_group" not in name and "bias" not in name:
                param.requires_grad = False

    def unfreeze_shared_parameters(self):
        """Unfreezes the shared parameters of the model (also called slow weights)."""
        for param in self.model.parameters():
            param.requires_grad = True

    def reset_rank1_scaling_factors(self):
        """Reinitializes the rank-1 scaling factors (also called fast weights)."""
        for module in self.model.modules():
            if hasattr(module, "r_group") and hasattr(module, "s_group"):
                module.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeats the input batch and passes it through the model."""
        repeat_shape = [self.num_estimators] + [1] * (x.dim() - 1)
        x = x.repeat(repeat_shape)
        return self.model(x)
