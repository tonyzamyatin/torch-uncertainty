import config
import lightning as pl
import torch
from torch import nn, optim

from torch_uncertainty.datamodules.classification.mnist import MNISTDataModule
from torch_uncertainty.models.lenet import batchensemble_lenet, lenet
from torch_uncertainty.models.wrappers.deep_ensembles import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)


if __name__ == "__main__":
    # Data
    dm = MNISTDataModule(
        root=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
        val_split=config.VAL_SPLIT,
    )

    # Models
    de = deep_ensembles(lenet(1, 10), num_estimators=config.NUM_ESTIMATORS, task="classification")
    be = batchensemble_lenet(in_channels=1, num_classes=10, num_estimators=config.NUM_ESTIMATORS)

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        fast_dev_run=config.FAST_DEV_RUN,
        max_epochs=config.MAX_EPOCHS,
        overfit_batches=config.OVERFIT_BATCHES,
    )
    for model in [be, de]:
        classifier = ClassificationRoutine(
            model=model,
            num_classes=10,
            loss=nn.CrossEntropyLoss(),
            is_ensemble=False,
            optim_recipe=optim.SGD(model.parameters(), lr=config.LEARING_RATE),
            eval_ood=config.EVAL_OOD,
            eval_shift=config.EVAL_SHIFT,
        )

        trainer.fit(classifier, dm)
        trainer.test(classifier, datamodule=dm)
