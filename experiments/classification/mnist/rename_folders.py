import os
import re

# Define the base directory
base_dir = "/workspace/experiments/classification/mnist/logs/lenet/batch_ensemble_retraining"

# Regular expression to match the folder names
pattern = re.compile(r"^(.*)_start_(\d+)_end_(\d+)$")

# Iterate over the folders in the base directory
for folder_name in os.listdir(base_dir):
    match = pattern.match(folder_name)
    if match:
        prefix, start_at, end_at = match.groups()
        new_folder_name = f"{prefix}_start_{int(start_at):02d}_end_{int(end_at):02d}"
        old_path = os.path.join(base_dir, folder_name)
        new_path = os.path.join(base_dir, new_folder_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{folder_name}' to '{new_folder_name}'")
