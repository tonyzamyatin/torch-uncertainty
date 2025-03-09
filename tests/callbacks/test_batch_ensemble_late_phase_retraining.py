from unittest.mock import MagicMock

import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from torch_uncertainty.callbacks.batch_ensemble_late_phase_retraining import (
    BatchEnsembleLatePhaseRetraining,
)
from torch_uncertainty.models.wrappers.batch_ensemble import BatchEnsemble


def test_on_setup_with_invalid_model():
    # Mock trainer and pl_module
    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.model = MagicMock()

    # Initialize callback
    callback = BatchEnsembleLatePhaseRetraining(start_epoch=2)

    # Test on_setup with invalid model
    with pytest.raises(TypeError):
        callback.setup(trainer, pl_module)


def test_on_setup_with_valid_model():
    # Mock trainer and pl_module
    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.model = MagicMock(spec=BatchEnsemble)

    # Initialize callback
    callback = BatchEnsembleLatePhaseRetraining(start_epoch=2)

    # Test on_setup with valid model
    callback.setup(trainer, pl_module)


def test_init_with_invalid_end_epoch():
    # Test with end_epoch <= start_epoch
    with pytest.raises(MisconfigurationException):
        BatchEnsembleLatePhaseRetraining(start_epoch=2, end_epoch=2)


def test_on_train_epoch_start():
    # Mock trainer and pl_module
    trainer = MagicMock()
    pl_module = MagicMock()
    pl_module.model = MagicMock(spec=BatchEnsemble)

    # Initialize callback with start_epoch
    callback = BatchEnsembleLatePhaseRetraining(start_epoch=2, end_epoch=4)

    # Assert that reset_rank1_scaling_factors and freeze_shared_parameters are not called before start_epoch
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, pl_module)
    pl_module.model.reset_rank1_scaling_factors.assert_not_called()
    pl_module.model.freeze_shared_parameters.assert_not_called()

    # After entering second phase, ensure calling on_train_epoch_start triggers reset
    trainer.current_epoch = 2
    callback.on_train_epoch_start(trainer, pl_module)
    pl_module.model.reset_rank1_scaling_factors.assert_called_once()
    pl_module.model.freeze_shared_parameters.assert_called_once()

    # After entering second phase, ensure calling on_train_epoch_start again does NOT re-trigger reset
    trainer.current_epoch = 3
    callback.on_train_epoch_start(trainer, pl_module)
    pl_module.model.reset_rank1_scaling_factors.assert_called_once()
    pl_module.model.freeze_shared_parameters.assert_called_once()

    # After entering third phase, ensure calling on_train_epoch_start triggers unfreeze
    trainer.current_epoch = 4
    callback.on_train_epoch_start(trainer, pl_module)
    pl_module.model.unfreeze_shared_parameters.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
