"""Object detection for satellite imagery using YOLOv8.

Uses YOLOv8n (nano) as a lightweight detector suitable for edge deployment.
Can detect vehicles, ships, aircraft, and other objects in satellite imagery.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from satellite_edge.models.base import BaseCVModel


class SatelliteDetector(BaseCVModel):
    """YOLOv8n-based object detector for satellite imagery.

    Uses pretrained YOLOv8n weights. For satellite-specific detection,
    fine-tune on datasets like DIOR or SpaceNet.

    Attributes:
        model: Underlying YOLO model instance.
        conf_threshold: Confidence threshold for detections.
    """

    name = "SatelliteDetector"
    task_type = 0  # DETECTION

    # YOLOv8n specs: 3.2M params, 8.7 GFLOPs = ~0.009 TOPS per inference
    _COMPUTE_COST_TOPS = 0.009

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        device: str = "cpu",
    ):
        """Initialize detector.

        Args:
            model_path: Path to YOLO weights. If None, uses pretrained yolov8n.
            conf_threshold: Confidence threshold for detections.
            device: Device to run inference on ('cpu', 'cuda', 'mps').
        """
        if model_path is None:
            # Use pretrained YOLOv8n
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(model_path))

        self.conf_threshold = conf_threshold
        self.device = device

        # COCO class names (subset relevant for satellite imagery)
        self._satellite_classes = {
            2: "car",
            5: "bus",
            7: "truck",
            8: "boat",
            4: "airplane",
        }

    def predict(self, tile: np.ndarray) -> dict[str, Any]:
        """Run object detection on a tile.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Dict with:
                - 'boxes': Array of [x1, y1, x2, y2] bounding boxes
                - 'scores': Array of confidence scores
                - 'classes': Array of class indices
                - 'class_names': List of class name strings
                - 'n_detections': Number of detections
                - 'inference_time_ms': Time taken for inference
        """
        # Ensure RGB format (YOLO expects RGB)
        if tile.shape[-1] == 4:
            tile = tile[:, :, :3]

        start_time = time.perf_counter()

        # Run inference
        results = self.model.predict(
            tile,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            return {
                "boxes": np.array([]).reshape(0, 4),
                "scores": np.array([]),
                "classes": np.array([], dtype=int),
                "class_names": [],
                "n_detections": 0,
                "inference_time_ms": inference_time_ms,
            }

        # Convert to numpy arrays
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Get class names
        class_names = [self.model.names.get(int(c), f"class_{c}") for c in classes]

        return {
            "boxes": xyxy,
            "scores": scores,
            "classes": classes,
            "class_names": class_names,
            "n_detections": len(boxes),
            "inference_time_ms": inference_time_ms,
        }

    def get_compute_cost(self) -> float:
        """Get compute cost in TOPS.

        YOLOv8n: ~8.7 GFLOPs = 0.0087 TOPS per inference.
        """
        return self._COMPUTE_COST_TOPS

    def get_memory_footprint(self) -> float:
        """Get memory footprint in MB.

        YOLOv8n: ~6.2 MB weights.
        """
        return 6.2

    def export_onnx(self, path: str | Path) -> Path:
        """Export model to ONNX format.

        Args:
            path: Output path for ONNX model.

        Returns:
            Path to exported model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.export(
            format="onnx",
            imgsz=384,
            simplify=True,
        )

        # YOLO exports to same directory as original model
        # Move to requested path if different
        return path

    def fine_tune(
        self,
        dataset_yaml: str | Path,
        epochs: int = 50,
        imgsz: int = 384,
        batch: int = 16,
        output_dir: str | Path = "runs/detect",
    ) -> Path:
        """Fine-tune on satellite imagery dataset.

        Args:
            dataset_yaml: Path to YOLO-format dataset configuration.
            epochs: Number of training epochs.
            imgsz: Input image size.
            batch: Batch size.
            output_dir: Output directory for training artifacts.

        Returns:
            Path to best weights.
        """
        results = self.model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(output_dir),
            device=self.device,
        )

        # Return path to best weights
        return Path(results.save_dir) / "weights" / "best.pt"

    def filter_satellite_classes(self, predictions: dict[str, Any]) -> dict[str, Any]:
        """Filter predictions to keep only satellite-relevant classes.

        Args:
            predictions: Output from predict().

        Returns:
            Filtered predictions with only vehicle/aircraft/boat classes.
        """
        if predictions["n_detections"] == 0:
            return predictions

        # Keep only satellite-relevant classes
        mask = np.isin(predictions["classes"], list(self._satellite_classes.keys()))

        return {
            "boxes": predictions["boxes"][mask],
            "scores": predictions["scores"][mask],
            "classes": predictions["classes"][mask],
            "class_names": [n for n, m in zip(predictions["class_names"], mask) if m],
            "n_detections": int(mask.sum()),
            "inference_time_ms": predictions["inference_time_ms"],
        }
