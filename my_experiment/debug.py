import config
import lightning as pl
import torch
from torch import nn, optim

import torch_uncertainty.datamodules.classification.mnist as tu_mnist
print(f"Imported MNISTDataModule from: {tu_mnist.__file__}")

from torch_uncertainty.models.lenet import batchensemble_lenet, lenet
from torch_uncertainty.models.wrappers.deep_ensembles import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)


if __name__ == "__main__":
    # Models
    be = batchensemble_lenet(in_channels=1, num_classes=10, num_estimators=config.NUM_ESTIMATORS)
    de = deep_ensembles(lenet(1, 10), num_estimators=config.NUM_ESTIMATORS, task="classification")

    # Data
    dm = tu_mnist.MNISTDataModule(
        root=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
        val_split=config.VAL_SPLIT,
    )

    # Trainer
    trainer = pl.Trainer(
            accelerator=config.ACCELERATOR,
            fast_dev_run=config.FAST_DEV_RUN,
            max_epochs=config.MAX_EPOCHS,
            overfit_batches=config.OVERFIT_BATCHES,
        )

    dm.prepare_data()
    dm.setup(stage="test")
    print(f"eval_shift is set to: {dm.eval_shift}")
    test_dls = dm.test_dataloader()

    for i, test_dl in enumerate(test_dls):
        print(f"Test Dataloader {i}: Number of batches = {len(test_dl)}")
        try:
            batch = next(iter(test_dl))
            x, y = batch
            print(f"Test Dataloader {i}: First batch shape = {x.shape}, Labels shape = {y.shape}")
        except StopIteration:
            print(f"Test Dataloader {i} is empty!")

        print()
        for model in [be, de]:
            # Run a forward pass to see if the model is producing outputs
            preds = model(x)
            print(f"{model.__class__.__name__} output shape: {preds.shape}")

    print("\n" + "="*50 + "\n")

    print("test_step() debug output:")
    classifier = ClassificationRoutine(
        model=de,
        num_classes=10,
        loss=nn.CrossEntropyLoss(),
        is_ensemble=False,
        optim_recipe=optim.SGD(de.parameters(), lr=config.LEARING_RATE),
        eval_ood=config.EVAL_OOD,
        eval_shift=config.EVAL_SHIFT,
    )
    trainer.test(classifier, datamodule=dm)

