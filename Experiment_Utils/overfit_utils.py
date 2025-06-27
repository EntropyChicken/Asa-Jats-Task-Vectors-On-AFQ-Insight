import torch
from torch.utils.data import DataLoader, TensorDataset

# This function is used to overfit a model to a small subset of the data.
# It is used to verify that a model can overfit a small subset of the data.

def make_overfit_loader(original_loader, n_samples: int = 32):
    """Return a DataLoader that repeatedly serves the *same* `n_samples`.

    The function fetches examples from the provided ``original_loader``
    until it has collected ``n_samples`` items, then wraps them in a
    ``TensorDataset`` and returns a new *shuffleable* ``DataLoader``.  This
    is handy when you want to verify that a network can **over-fit** a very
    small slice of the dataset – if the model cannot drive the loss close
    to zero on this miniature problem, the architecture / likelihood is
    probably to blame rather than the optimisation schedule.

    Parameters
    ----------
    original_loader : torch.utils.data.DataLoader
        Any DataLoader that yields ``(x, y)`` batches.
    n_samples : int, optional
        Total number of samples to keep.  Defaults to 32.

    Returns
    -------
    torch.utils.data.DataLoader
        A loader that serves the fixed subset, using ``batch_size=n_samples``
        so the whole subset is seen each iteration.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    xs, ys = [], []
    collected = 0

    # We iterate through the original loader until we have enough samples.
    for batch_x, batch_y in original_loader:
        remaining = n_samples - collected
        if remaining <= 0:
            break

        # Trim the batch if it would overshoot.
        take = min(remaining, batch_x.size(0))
        xs.append(batch_x[:take])
        ys.append(batch_y[:take])
        collected += take

        if collected >= n_samples:
            break

    if collected < n_samples:
        raise RuntimeError(
            f"Requested {n_samples} samples but original_loader only provided "
            f"{collected}. Consider lowering n_samples or using a larger loader.")

    x_fixed = torch.cat(xs, dim=0)
    y_fixed = torch.cat(ys, dim=0)

    dataset_fixed = TensorDataset(x_fixed, y_fixed)

    # One batch per epoch keeps things deterministic; shuffle=True so order
    # changes between epochs (not critical for over-fit test).
    return DataLoader(dataset_fixed, batch_size=n_samples, shuffle=True) 