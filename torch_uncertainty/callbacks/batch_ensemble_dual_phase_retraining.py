import lightning.pytorch as pl


class BatchEnsembleDualPhaseRetraining(pl.Callback):
    """Dual retraining for BatchEnsembles based on convergence.

    This callback:
    1. Trains all parameters until convergence.
    2. Freezes the shared weight matrix and resets rank-1 scaling factors.
    3. Retrains only rank-1 scaling factors until convergence.
    4. Repeats for a given number of cycles.

    Args:
        cycles (int): Number of retraining cycles.
        convergence_patience (int, optional): Number of epochs without improvement before considering convergence. Defaults to 5.
        monitor (str, optional): Metric to monitor. Defaults to "val_loss".
        mode (str, optional): "min" for loss, "max" for accuracy. Defaults to "min".
    """
