import pytorch_lightning as pl

from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble


class BatchEnsembleLatePhaseTraining(pl.Callback):
    """Callback for training a BatchEnsemble model in two phases.
    1. Training with all parameters active.
    2. Training with shared parameters frozen and rank-1 scaling factors reset.
    3. Optionally, unfreeze shared parameters.

    This callback controls when the shared parameters of a BatchEnsemble model are frozen and when
    the rank-1 scaling factors are reset, based on predefined epochs or convergence criteria.

    Args:
        start_epoch (int, optional): Epoch to start freezing shared parameters. Defaults to None.
        end_epoch (int, optional): Epoch to unfreeze shared parameters. Defaults to None.
        start_at_convergence (bool, optional): Start freezing shared parameters when model has converged. Defaults to None.
        end_at_convergence (bool, optional): Unfreeze shared parameters when model has converged. Defaults to None.
        convergence_patience (int, optional): Number of epochs without improvement before considering convergence. Defaults to 5.
        monitor (str, optional): Metric to monitor. Defaults to "val_loss".
        mode (str, optional): "min" for loss, "max" for accuracy. Defaults to "min".

    Raises:
        ValueError: If conflicting arguments are provided, i.e.
        - if not *exactly* one of `start_epoch` or `start_at_convergence` is set, or
        - if *both* `end_epoch` and `end_at_convergence` are set.
    """

    def __init__(
        self,
        start_epoch: int | None = None,
        end_epoch: int | None = None,
        start_at_convergence: bool | None = None,
        end_at_convergence: bool | None = None,
        convergence_patience: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        super().__init__()
        if (start_epoch is None and start_at_convergence is None) or (
            start_epoch is not None and start_at_convergence is not None
        ):
            raise ValueError(
                "Either `start_epoch` or `start_at_convergence` must be set, but not both."
            )
        if end_epoch is not None and end_at_convergence is not None:
            raise ValueError("Either `end_epoch` or `end_at_convergence` can be set, but not both.")

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_at_convergence = start_at_convergence
        self.end_at_convergence = end_at_convergence
        self.convergence_patience = convergence_patience
        self.monitor = monitor
        self.mode = mode

        # Track metric history for detecting convergence
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.converged = False

    def on_setup(self, trainer, pl_module) -> None:
        if not isinstance(pl_module.model, BatchEnsemble):
            raise TypeError("The model must be an instance of BatchEnsemble.")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # If using start_epoch, apply freezing at the specified epoch
        if self.start_epoch is not None and trainer.current_epoch == self.start_epoch:
            pl_module.model.reset_rank1_scaling_factors()
            pl_module.model.freeze_shared_parameters()

        # If using convergence, check if model has converged and apply freezing
        if self.start_at_convergence and self.converged:
            pl_module.model.reset_rank1_scaling_factors()
            pl_module.model.freeze_shared_parameters()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Track the monitored metric (e.g., validation loss or accuracy)
        current_metric = trainer.callback_metrics.get(self.monitor)

        if current_metric is not None:
            if (self.mode == "min" and current_metric < self.best_metric) or (
                self.mode == "max" and current_metric > self.best_metric
            ):
                self.best_metric = current_metric
                self.wait = 0  # Reset patience counter
            else:
                self.wait += 1

            # If patience exceeded, model has converged
            if self.wait >= self.convergence_patience:
                self.converged = True

        # If using end_epoch, unfreeze at specified epoch
        if self.end_epoch is not None and trainer.current_epoch == self.end_epoch:
            pl_module.model.unfreeze_shared_parameters()

        # If using end_at_convergence, unfreeze when model has converged
        if self.end_at_convergence and self.converged:
            pl_module.model.unfreeze_shared_parameters()
