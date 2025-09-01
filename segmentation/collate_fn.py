import torch
import numpy as np

def collate_fn(batch, npoints=32768):
    """
    Custom collate function to pad all point clouds in a batch to `npoints`.

    Parameters:
        batch (list): List of tuples (points, label, target, centroid, m) for each sample.
        npoints (int): The number of points to pad/truncate to (default: 32768).

    Returns:
        tuple: (points, labels, targets, centroids, scales)
    """
    padded_points = []
    padded_targets = []
    labels = []

    for points, label, target in batch:
        # If the point cloud has fewer points than npoints, pad it
        if points.shape[0] < npoints:
            padding = np.zeros((npoints - points.shape[0], points.shape[1]))
            padded_points.append(np.vstack((points, padding)))
            padded_targets.append(np.hstack((target, np.full((npoints - len(target)), -99999))))  # Fill with -1 for padding
        else:
            # Otherwise, truncate to npoints
            padded_points.append(points[:npoints, :])
            padded_targets.append(target[:npoints])

        labels.append(label)

    # Convert to torch tensors
    padded_points = torch.tensor(np.stack(padded_points), dtype=torch.float32)
    padded_targets = torch.tensor(np.stack(padded_targets), dtype=torch.long)
    labels = torch.tensor(np.stack(labels), dtype=torch.long)


    return padded_points, labels, padded_targets
