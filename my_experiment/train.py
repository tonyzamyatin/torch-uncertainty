import lightning as pl
from torch import nn, optim

from torch_uncertainty.datamodules.classification.mnist import MNISTDataModule
from torch_uncertainty.models.lenet import batchensemble_lenet, lenet
from torch_uncertainty.models.wrappers.deep_ensembles import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine

pl.seed_everything(42)

if __name__ == "__main__":
    # Data
    dm = MNISTDataModule(root="data", batch_size=32, eval_ood=True, eval_shift=True, val_split=0.2)

    # Models
    be = batchensemble_lenet(in_channels=1, num_classes=10, num_estimators=4)

    de = deep_ensembles(lenet(1, 10), num_estimators=4, task="classification")

    trainer = pl.Trainer(accelerator="gpu", fast_dev_run=True, max_epochs=5, overfit_batches=1)
    for model in [be, de]:
        classifier = ClassificationRoutine(
            model=model,
            num_classes=10,
            loss=nn.CrossEntropyLoss(),
            is_ensemble=True,
            optim_recipe=optim.SGD(model.parameters(), lr=0.01),
            eval_ood=True,
            eval_shift=True,
        )

        trainer.fit(classifier, dm)
        trainer.test(classifier, datamodule=dm)
