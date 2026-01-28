"""Value estimation for satellite imagery tiles.

Computes metrics to estimate the "value" of processing a tile for different
task types: cloud coverage, information entropy, and edge density.
"""

import numpy as np
from skimage.filters import sobel
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk


class ValueEstimator:
    """Estimate the value of satellite image tiles for scheduling.

    Computes three metrics:
    - Cloud fraction: Proportion of tile covered by clouds (brightness-based)
    - Entropy: Information content / texture complexity
    - Edge density: Feature richness (potential detection targets)

    These metrics help the RL scheduler prioritize tiles based on content.
    """

    def __init__(
        self,
        cloud_threshold_percentile: float = 85.0,
        entropy_disk_size: int = 5,
    ):
        """Initialize value estimator.

        Args:
            cloud_threshold_percentile: Percentile for cloud detection threshold.
                Higher values are more conservative (fewer cloud pixels).
            entropy_disk_size: Radius of disk for local entropy calculation.
        """
        self.cloud_threshold_percentile = cloud_threshold_percentile
        self.entropy_disk_size = entropy_disk_size

    def estimate_cloud_fraction(self, tile: np.ndarray) -> float:
        """Estimate fraction of tile covered by clouds.

        Uses simple brightness thresholding. Clouds appear bright in visible
        bands. This is a fast heuristic, not a full cloud detection model.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Cloud fraction in [0, 1].
        """
        if tile.size == 0:
            return 0.0

        # Convert to grayscale by averaging channels
        if tile.ndim == 3:
            gray = np.mean(tile, axis=-1)
        else:
            gray = tile

        # Normalize to [0, 1] if needed
        if gray.max() > 1.0:
            gray = gray / gray.max() if gray.max() > 0 else gray

        # Use percentile-based threshold (clouds are bright)
        threshold = np.percentile(gray, self.cloud_threshold_percentile)

        # Count pixels above threshold as clouds
        cloud_mask = gray > threshold
        return float(cloud_mask.mean())

    def compute_entropy(self, tile: np.ndarray) -> float:
        """Compute normalized Shannon entropy of the tile.

        Higher entropy indicates more texture/information content.
        Uniform areas (ocean, desert) have low entropy.
        Urban areas, forests have high entropy.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Normalized entropy in [0, 1].
        """
        if tile.size == 0:
            return 0.0

        # Convert to grayscale
        if tile.ndim == 3:
            gray = np.mean(tile, axis=-1)
        else:
            gray = tile.copy()

        # Normalize to uint8 for entropy calculation
        if gray.max() > 1.0:
            gray = (gray / gray.max() * 255).astype(np.uint8)
        else:
            gray = (gray * 255).astype(np.uint8)

        # Handle edge case of constant image
        if gray.max() == gray.min():
            return 0.0

        # Local entropy with disk structuring element
        try:
            ent = rank_entropy(gray, disk(self.entropy_disk_size))
            # Normalize by max possible entropy (8 bits)
            return float(ent.mean() / 8.0)
        except Exception:
            # Fallback: histogram-based global entropy
            hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 255))
            hist = hist[hist > 0]
            prob = hist / hist.sum()
            return float(-np.sum(prob * np.log2(prob)) / 8.0)

    def compute_edge_density(self, tile: np.ndarray) -> float:
        """Compute edge density as proxy for feature richness.

        High edge density suggests many potential detection targets
        (buildings, vehicles, ships, etc.).

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Normalized edge density in [0, 1].
        """
        if tile.size == 0:
            return 0.0

        # Convert to grayscale
        if tile.ndim == 3:
            gray = np.mean(tile, axis=-1)
        else:
            gray = tile

        # Normalize to [0, 1]
        if gray.max() > 1.0:
            gray = gray / gray.max() if gray.max() > 0 else gray

        # Sobel edge detection
        edges = sobel(gray)

        # Use 95th percentile as robust measure of edge strength
        # Normalize by typical max edge response (~0.5 for strong edges)
        edge_strength = np.percentile(edges, 95)
        return float(min(edge_strength / 0.5, 1.0))

    def compute_all_metrics(self, tile: np.ndarray) -> dict[str, float]:
        """Compute all value metrics for a tile.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Dict with 'cloud_fraction', 'entropy', 'edge_density' keys.
        """
        return {
            "cloud_fraction": self.estimate_cloud_fraction(tile),
            "entropy": self.compute_entropy(tile),
            "edge_density": self.compute_edge_density(tile),
        }

    def compute_task_value(
        self,
        tile: np.ndarray,
        task_type: int,
        base_value: float = 1.0,
    ) -> float:
        """Compute task-specific value for a tile.

        Different task types benefit from different image characteristics:
        - DETECTION (0): Penalize clouds, reward features
        - ANOMALY (1): Penalize clouds, reward entropy + features
        - CLOUD_MASK (2): Reward clouds (more to mask)
        - COMPRESSION (3): Reward entropy (better compression)

        Args:
            tile: Array of shape (H, W, C) with pixel values.
            task_type: Integer task type (0-3).
            base_value: Base value to multiply.

        Returns:
            Task-specific value.
        """
        metrics = self.compute_all_metrics(tile)
        cloud = metrics["cloud_fraction"]
        entropy = metrics["entropy"]
        edges = metrics["edge_density"]

        if task_type == 0:  # DETECTION
            # Cloudy images are useless for detection
            multiplier = (1 - cloud) * (0.3 * entropy + 0.7 * edges)
        elif task_type == 1:  # ANOMALY
            # Similar to detection but entropy matters more
            multiplier = (1 - cloud) * (0.5 * entropy + 0.5 * edges)
        elif task_type == 2:  # CLOUD_MASK
            # More clouds = more value for cloud masking task
            multiplier = 0.2 + 0.8 * cloud  # Minimum 0.2 for clear sky
        else:  # COMPRESSION (3)
            # Higher entropy = better compression candidate
            multiplier = 0.3 + 0.7 * entropy  # Minimum 0.3

        return base_value * max(multiplier, 0.1)  # Floor at 10% of base
