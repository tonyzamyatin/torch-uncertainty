import lightning.pytorch as pl
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing_extensions import override

from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble


class BatchEnsembleLatePhaseRetraining(pl.Callback):
    """Late-phase retraining for BatchEnsembles.
    This callback:
    1. Trains all parameters until the :arg:`start_epoch` is reached.
    2. Freezes the shared weight matrix and resets rank-1 scaling factors.
    3. Retrains the rank-1 scaling factors.
    4. Optionally, unfreeze shared parameters and continue training all parameters when :arg:`end_epoch' is reached.

    Args:
        start_epoch (int): Epoch to start freezing shared parameters. Defaults to None.
        end_epoch (int, optional): Epoch to unfreeze shared parameters. Defaults to None.

    Raises:
        MisconfigurationException: If :arg:`end_epoch` is not greater than :arg:`start_epoch`.
        TypeError: If the model is not an instance of :class:`BatchEnsemble` wrapper.
    """

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int | None = None,
    ) -> None:
        super().__init__()
        if end_epoch and end_epoch <= start_epoch:
            raise MisconfigurationException("`end_epoch` must be greater than `start_epoch`.")
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if not hasattr(pl_module, "model"):
            raise AttributeError("The provided LightningModule does not have a `model` attribute.")
        if not isinstance(pl_module.model, BatchEnsemble):
            raise TypeError(
                "The `model` attribute of LightningModule must be an instance of BatchEnsemble."
            )

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch == self.start_epoch:
            pl_module.model.reset_rank1_scaling_factors()
            pl_module.model.freeze_shared_parameters()
        elif self.end_epoch and trainer.current_epoch == self.end_epoch:
            pl_module.model.unfreeze_shared_parameters()
