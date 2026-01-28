"""CV models for satellite image processing.

Lightweight models designed for edge deployment:
- SatelliteDetector: YOLOv8n-based object detection
- CloudMaskModel: Lightweight U-Net for cloud segmentation
- ImageCompressor: Convolutional autoencoder for compression
"""

from satellite_edge.models.base import BaseCVModel
from satellite_edge.models.cloud_mask import CloudMaskModel
from satellite_edge.models.compressor import ImageCompressor
from satellite_edge.models.detector import SatelliteDetector

__all__ = [
    "BaseCVModel",
    "SatelliteDetector",
    "CloudMaskModel",
    "ImageCompressor",
]
