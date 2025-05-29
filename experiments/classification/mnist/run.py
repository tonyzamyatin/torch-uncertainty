import subprocess

architecture = "mlp"
num_estimators = 4

losses = ["ce", "repulsive"]
versions = ["batched"]
ranks = [1, "full"]
lrs = [1e-3]
seeds = [42, 69, 123, 456, 789]

for loss in losses:
    for version in versions:
        if version == "standard" and loss == "repulsive":
            continue
        for i, rank in enumerate(ranks):
            if version != "batched" and i > 0:
                continue
            for lr in lrs:
                for seed in seeds:
                    save_dir = f"logs/{architecture}/{loss}_adam_lr={lr}"
                    name = version if version != "batched" else f"{version}-rank-{rank}"

                    cmd = [
                        "python",
                        "cli.py",
                        "fit",
                        "-c",
                        f"configs/{architecture}/{version}.yaml",
                        "--trainer.fast_dev_run=True",
                        "--trainer.accelerator=gpu",
                        f"--optimizer.lr={lr}",
                        f"--seed_everything={seed}",
                        f"--trainer.logger.init_args.save_dir={save_dir}",
                        f"--trainer.logger.init_args.name={name}",
                        f"--trainer.logger.init_args.version=seed_{seed}",
                        "--model.model.init_args.gradient_blocking=True",
                    ]

                    if version != "standard":
                        cmd.append(f"--model.model.init_args.num_estimators={num_estimators}")
                        cmd.append(
                            f"--model.format_batch_fn.init_args.num_repeats={num_estimators}"
                        )
                        if loss == "repulsive":
                            cmd.append("--model.loss=RepulsiveCrossEntropyLoss")
                            (cmd.append(f"--model.loss.init_args.num_estimators={num_estimators}"),)

                    if version == "batched":
                        cmd.append(f"--model.model.rank={rank}")

                    subprocess.run(cmd)
