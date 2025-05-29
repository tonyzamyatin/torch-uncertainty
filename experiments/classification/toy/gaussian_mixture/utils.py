import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, loss_fn, optimizer, epochs, train_loader, val_loader, format_batch_fn):
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    # Train the model
    for i, epoch in enumerate(progress_bar):
        print_debug = False
        if print_debug:
            print("DEBUG: Sanity check")
        train_step(
            model,
            optimizer,
            loss_fn,
            train_loader,
            format_batch_fn,
            device,
            print_debug=print_debug,
        )
        val_loss, val_acc = val_step(model, val_loader, format_batch_fn, device)
        # add val_loss and val_acc to progress bar
        progress_bar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.2f}%")


def train_step(model, optimizer, loss_fn, data_loader, format_batch_fn, device, print_debug=False):
    model.train()
    for i, batch in enumerate(data_loader):
        if print_debug:
            print(f"DEBUG: batch {i}")
        inputs, target = format_batch_fn(batch)

        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        if print_debug:
            print(f"DEBUG: input: {inputs.shape}, output: {output.shape}, target: {target.shape}")

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


def val_step(model, data_loader, format_batch_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, target = format_batch_fn(batch)
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)

            loss = F.cross_entropy(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()

    return total_loss / len(data_loader), correct / len(data_loader.dataset)


def predict(model, data_loader, format_batch_fn):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, _ = format_batch_fn(batch)
            output = model(inputs)
            predictions.append(output)
    return torch.cat(predictions)


def model_uncertainty_surface(
    model,
    grid_limits,
    grid_res=200,
    criterion="entropy",
    device="cpu",
    print_debug=False,
):
    """Compute the uncertainty surface for a given model.

    Args:
        model (nn.Module): The model to evaluate.
        grid_limits (tuple): (xmin, xmax) = (ymin, ymax).
        grid_res (int): Resolution of the grid.
        criterion (str): Uncertainty measure ('entropy', 'mi', 'disagreement').
        device (str): Device to run the computation on.

    Returns:
        np.ndarray: The uncertainty surface.
    """
    assert criterion in [
        "entropy",
        "mi",
        "disagreement",
    ], f"Unsupported criterion '{criterion}'. Choose from ['entropy', 'mi', 'disagreement']."

    # Create the grid
    x = torch.linspace(grid_limits[0], grid_limits[1], grid_res)
    y = torch.linspace(grid_limits[0], grid_limits[1], grid_res)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=-1).to(device)

    if print_debug:
        print(f"DEBUG: grid shape: {grid.shape}")

    # Compute logits
    logits = model(grid)  # (M * B, C)
    if print_debug:
        print(f"DEBUG: logits shape: {logits.shape}")

    # Reshape logits to (M, B, C)
    logits = logits.view(-1, grid_res * grid_res, logits.shape[-1])  # (M, B, C)

    # Handle invalid logits
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute probabilities
    probs = F.softmax(logits, dim=-1)  # (M, B, C)

    # Clamp probabilities to avoid log(0)
    probs = probs.clamp(min=1e-9)

    # Compute mean probabilities
    mean_probs = probs.mean(dim=0)  # (B, C)

    # Compute uncertainty based on the criterion
    if criterion == "entropy":
        uncertainty = -(mean_probs * mean_probs.log()).sum(dim=-1)
    elif criterion == "mi":
        entropy_mean = -(mean_probs * mean_probs.log()).sum(dim=-1)
        mean_entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=0)
        uncertainty = entropy_mean - mean_entropy
    elif criterion == "disagreement":
        pred_labels = probs.argmax(dim=-1)  # shape (M, B)
        mode_labels = pred_labels.mode(dim=0).values  # (B,)
        disagreement = (pred_labels != mode_labels.unsqueeze(0)).float().mean(dim=0)
        uncertainty = disagreement
    else:
        raise NotImplementedError

    # Reshape uncertainty to grid resolution
    uncertainty = uncertainty.view(grid_res, grid_res).detach().cpu().numpy()
    if print_debug:
        print(f"DEBUG: uncertainty shape: {uncertainty.shape}")

    return uncertainty


def plot_uncertainty_surfaces(
    uncertainty_surfaces,
    row_names,
    col_names,
    dataset,
    grid_limits,
    criterion: str = None,
    uncertainty_cmap="viridis",
    save_path: str = None,
    normalize: bool = True,
):
    label_size = 18
    title_size = 28
    pad = 20

    n_rows = len(uncertainty_surfaces)
    n_cols = len(uncertainty_surfaces[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    X = dataset[:][0]
    X = X.cpu().numpy()

    if normalize:
        all_vals = np.concatenate([np.ravel(u) for row in uncertainty_surfaces for u in row])
        vmin, vmax = np.min(all_vals), np.max(all_vals)
    else:
        vmin = vmax = None

    axes = np.atleast_2d(axes)
    im = None

    for i, (row, row_name) in enumerate(zip(uncertainty_surfaces, row_names, strict=False)):
        for j, (uncertainty, col_name) in enumerate(zip(row, col_names, strict=False)):
            ax = axes[i, j]
            im = ax.imshow(
                uncertainty,
                extent=[*grid_limits, *grid_limits],
                origin="lower",
                cmap=uncertainty_cmap,
                interpolation="bilinear",
                alpha=0.85,
                vmin=vmin,
                vmax=vmax,
            )
            ax.scatter(
                X[:, 0], X[:, 1], c=dataset.targets, cmap="Set3", edgecolor="k", s=10, alpha=0.6
            )
            if i == 0:
                ax.set_title(col_name, fontsize=label_size, pad=pad)
            if j == 0:
                ax.set_ylabel(
                    row_name,
                    fontsize=label_size,
                    rotation=90,
                    labelpad=pad,
                    va="center",
                    ha="center",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    # Leave space at the right for colorbar and at the top for title
    # plt.tight_layout(rect=[0, 0, 0.93, 0.90])

    # Set equal spacing between columns and rows
    fig.subplots_adjust(left=0.07, right=0.92, top=0.83, bottom=0.03, wspace=0.1, hspace=0.1)

    # Add colorbar in a new axis to the right of the plots
    cbar_ax = fig.add_axes([0.94, 0.025, 0.015, 0.8])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar_label = "Uncertainty" if criterion is None else criterion.capitalize()
    # cbar.set_label(cbar_label, fontsize=label_size, labelpad=pad)

    title_str = (
        f"Uncertainty Surfaces ({criterion.capitalize()})" if criterion else "Uncertainty Surfaces"
    )
    fig.suptitle(title_str, fontsize=title_size)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
