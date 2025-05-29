import json
import subprocess

# Define base callbacks
base_callbacks = [
    {
        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
        "init_args": {
            "monitor": "val/cls/Acc",  # Ensure this is the correct metric
            "mode": "max",
            "save_top_k": 1,  # Save the best checkpoint
            "save_last": True,  # Also save the last checkpoint
            "filename": "best",  # Ensure the best checkpoint gets named correctly
        },
    },
    {
        "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
        "init_args": {"logging_interval": "step"},
    },
    {
        "class_path": "lightning.pytorch.callbacks.EarlyStopping",
        "init_args": {"monitor": "val/cls/Acc", "patience": 1000, "check_finite": True},
    },
]

# List of seeds to use for different runs
seeds = [42, 69, 123, 456, 789]

# Run the resnet.py script with different seeds and end_epoch values
for version in ["ll-ensemble"]:
    for seed in seeds:
        for rank in [1]:
            # Modify BatchEnsembleLatePhaseRetraining entry
            # callbacks = base_callbacks + [{
            #     "class_path": "torch_uncertainty.callbacks.BatchEnsembleLatePhaseRetraining",
            #     "init_args": {"start_epoch": start, "end_epoch": 75, "last_layer_only": False}
            # }]
            command = [
                "python",
                "resnet.py",
                "fit",
                "-c",
                f"configs/resnet18/{version}.yaml",
                # "--ckpt_path", f"logs/resnet18/{version}/version_0/checkpoints/last.ckpt",
                # f"--trainer.fast_dev_run=True",
                # f"--model.num_estimators=1",
                # f"--model.rank={rank}",
                "--trainer.accelerator=gpu",
                f"--trainer.logger.init_args.name={version}",
                f"--trainer.logger.init_args.version=seed_{seed}",
                f"--trainer.callbacks={json.dumps(base_callbacks)}",
                f"--seed_everything={seed}",  # Pass the seed to the CLI
            ]

            print("Executing:", " ".join(command))  # Debugging print

            subprocess.run(command)
