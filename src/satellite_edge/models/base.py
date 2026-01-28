"""Base class for CV models.

Defines the interface that all CV models must implement for integration
with the satellite edge scheduler.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseCVModel(ABC):
    """Abstract base class for satellite CV models.

    All models must implement predict() and get_compute_cost().
    Models should be lightweight enough for edge deployment.

    Attributes:
        name: Human-readable model name.
        task_type: TaskType enum value this model handles.
    """

    name: str = "BaseModel"
    task_type: int = -1  # Override in subclasses

    @abstractmethod
    def predict(self, tile: np.ndarray) -> dict[str, Any]:
        """Run inference on a tile.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Dict with model-specific outputs. All models should include:
                - 'inference_time_ms': Time taken for inference
                - Model-specific outputs (detections, masks, etc.)
        """
        pass

    @abstractmethod
    def get_compute_cost(self) -> float:
        """Get estimated compute cost in TOPS.

        Returns:
            Estimated tera-operations for processing one tile.
        """
        pass

    def get_memory_footprint(self) -> float:
        """Get model memory footprint in MB.

        Returns:
            Estimated memory usage including weights and activations.
        """
        return 0.0  # Override in subclasses

    def export_onnx(self, path: str | Path) -> Path:
        """Export model to ONNX format.

        Args:
            path: Output path for ONNX model.

        Returns:
            Path to exported model.

        Raises:
            NotImplementedError: If export is not supported.
        """
        raise NotImplementedError(f"{self.name} does not support ONNX export")

    def warmup(self, tile_shape: tuple[int, int, int] = (384, 384, 4)) -> None:
        """Warm up the model with a dummy inference.

        Useful for getting accurate timing on subsequent calls.

        Args:
            tile_shape: Shape of dummy tile (H, W, C).
        """
        dummy = np.zeros(tile_shape, dtype=np.uint8)
        self.predict(dummy)
