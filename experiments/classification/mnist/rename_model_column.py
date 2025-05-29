import pandas as pd

for set in ["test", "shift", "ood"]:
    df = pd.read_csv(f"results/mlp/gb_repulsive_adam_lr=0.001/all/{set}_metrics.csv")
    # Prepend "gb-" prefix to "Name" columns
    df["Name"] = "gb-" + df["Name"]
    df.to_csv(f"results/mlp/gb_repulsive_adam_lr=0.001/all/{set}_metrics.csv", index=False)
