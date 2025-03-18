from typing import Tuple

import torch


def kmeans_plus_plus(
    X: torch.Tensor,
    n_clusters: int,
    device: torch.device = None,
    random_state: int = None,
) -> torch.Tensor:
    """
    Initialize cluster centers using k-means++

    Args:
        X: Data tensor of shape (n_samples, n_dimensions)
        n_clusters: Number of clusters
        device: Device to use for computation
        random_state: Random seed for reproducibility

    Returns:
        Cluster centers of shape (n_clusters, n_dimensions)
    """
    if device is None:
        device = X.device

    # Set random seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples, n_features = X.shape

    # Choose the first center randomly
    indices = torch.randperm(n_samples, device=device)[:1]
    centers = X[indices]

    # Choose remaining centers
    for _ in range(1, n_clusters):
        # Compute squared distances to nearest center
        distances = torch.cdist(X, centers, p=2.0)
        min_distances, _ = torch.min(distances, dim=1)

        # Square distances to convert to probabilities
        squared_distances = min_distances**2

        # Choose next center with probability proportional to squared distance
        probs = squared_distances / squared_distances.sum()
        cumprobs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=device)
        ind = torch.searchsorted(cumprobs, r)

        # Add new center
        centers = torch.cat([centers, X[ind : ind + 1]], dim=0)

    return centers


def kmeans(
    X: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 1e-4,
    device: torch.device = None,
    random_state: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means clustering

    Args:
        X: Data tensor of shape (n_samples, n_dimensions)
        n_clusters: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        device: Device to use for computation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cluster_centers, labels)
    """
    if device is None:
        device = X.device

    # Initialize centers using k-means++
    centers = kmeans_plus_plus(X, n_clusters, device, random_state)

    prev_inertia = float("inf")

    for _ in range(max_iters):
        # Assign samples to nearest center
        distances = torch.cdist(X, centers, p=2.0)
        labels = torch.argmin(distances, dim=1)

        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(n_clusters):
            # Get points in cluster k
            cluster_mask = labels == k

            if cluster_mask.sum() > 0:
                # Update center as mean of points
                new_centers[k] = X[cluster_mask].mean(dim=0)
            else:
                # If no points, keep old center
                new_centers[k] = centers[k]

        # Check for convergence
        inertia = torch.sum(torch.min(distances, dim=1)[0])
        centers = new_centers

        if abs(prev_inertia - inertia) < tol:
            break

        prev_inertia = inertia

    return centers, labels


def minibatch_kmeans(
    X: torch.Tensor,
    n_clusters: int,
    batch_size: int = 1024,
    max_iters: int = 100,
    tol: float = 1e-4,
    device: torch.device = None,
    random_state: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mini-batch K-means clustering

    Args:
        X: Data tensor of shape (n_samples, n_dimensions)
        n_clusters: Number of clusters
        batch_size: Mini-batch size
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
        device: Device to use for computation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cluster_centers, labels)
    """
    if device is None:
        device = X.device

    # Set random seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples = X.shape[0]

    # Initialize centers using k-means++
    centers = kmeans_plus_plus(X, n_clusters, device, random_state)

    # Initialize counters for each center
    counts = torch.ones(n_clusters, device=device)

    prev_centers_norm = centers.norm(dim=1).sum()

    for i in range(max_iters):
        # Sample a mini-batch
        batch_indices = torch.randperm(n_samples, device=device)[:batch_size]
        batch = X[batch_indices]

        # Assign samples to nearest center
        distances = torch.cdist(batch, centers, p=2.0)
        labels = torch.argmin(distances, dim=1)

        # Update centers
        for k in range(n_clusters):
            # Get points in cluster k
            cluster_mask = labels == k

            if cluster_mask.sum() > 0:
                # Update counts
                counts[k] += cluster_mask.sum()

                # Compute step size
                eta = 1.0 / counts[k]

                # Update center
                centers[k] = (1 - eta) * centers[k] + eta * batch[cluster_mask].mean(
                    dim=0
                )

        # Check for convergence
        centers_norm = centers.norm(dim=1).sum()

        if abs(prev_centers_norm - centers_norm) < tol:
            break

        prev_centers_norm = centers_norm

    # Assign all samples to final centers
    distances = torch.cdist(X, centers, p=2.0)
    labels = torch.argmin(distances, dim=1)

    return centers, labels
