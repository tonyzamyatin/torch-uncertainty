import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianMixtureDataset(Dataset):
    def __init__(self, n_samples_per_class=100, std=0.1, radius=0.7, n_classes=5, seed=42):
        np.random.seed(seed)
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        self.centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        self.n_classes = len(self.centers)
        self.data = []
        self.targets = []
        for i, center in enumerate(self.centers):
            points = np.random.randn(n_samples_per_class, 2) * std + center
            self.data.append(points)
            self.targets.append(np.full(n_samples_per_class, i))
        self.data = torch.tensor(np.vstack(self.data), dtype=torch.float32)
        self.targets = torch.tensor(np.hstack(self.targets), dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == "__main__":
    # plot the dataset
    import matplotlib.pyplot as plt

    dataset = GaussianMixtureDataset()
    data, targets = dataset.data.numpy(), dataset.targets.numpy()
    plt.scatter(data[:, 0], data[:, 1], c=targets, cmap="viridis", s=10)
    plt.title("Gaussian Mixture Toy Dataset")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig("gaussian_mixture_toy_dataset.png")  # Save the plot to a file
    print("Plot saved as 'gaussian_mixture_toy_dataset.png'")
