"""Cloud mask segmentation model.

Lightweight U-Net architecture for cloud detection in satellite imagery.
Designed for edge deployment with minimal compute requirements.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from satellite_edge.models.base import BaseCVModel


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient computation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class LightweightUNet(nn.Module):
    """Lightweight U-Net for cloud segmentation.

    Uses depthwise separable convolutions to reduce compute.
    ~12% of standard U-Net FLOPs while maintaining accuracy.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            DepthwiseSeparableConv(256, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(64, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class CloudMaskModel(BaseCVModel):
    """Cloud mask model for satellite imagery.

    Uses a lightweight U-Net to segment clouds in RGB(NIR) imagery.
    Output is a binary mask where 1 = cloud, 0 = clear.

    Attributes:
        model: Underlying PyTorch model.
        device: Device for inference.
        threshold: Threshold for binary mask.
    """

    name = "CloudMaskModel"
    task_type = 2  # CLOUD_MASK

    # Lightweight U-Net: ~1 GFLOPs = 0.001 TOPS per inference
    _COMPUTE_COST_TOPS = 0.001

    def __init__(
        self,
        in_channels: int = 4,
        weights_path: str | Path | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        """Initialize cloud mask model.

        Args:
            in_channels: Number of input channels (3=RGB, 4=RGBNIR).
            weights_path: Path to pretrained weights. If None, uses random init.
            device: Device for inference.
            threshold: Threshold for converting probabilities to binary mask.
        """
        self.device = device
        self.threshold = threshold
        self.in_channels = in_channels

        self.model = LightweightUNet(in_channels=in_channels, out_channels=1)

        if weights_path is not None:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )

        self.model.to(device)
        self.model.eval()

    def predict(self, tile: np.ndarray) -> dict[str, Any]:
        """Run cloud segmentation on a tile.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Dict with:
                - 'mask': Binary mask array (H, W) where 1=cloud
                - 'probabilities': Probability map (H, W) in [0, 1]
                - 'cloud_fraction': Fraction of tile covered by clouds
                - 'inference_time_ms': Time taken for inference
        """
        # Handle channel mismatch
        if tile.shape[-1] != self.in_channels:
            if tile.shape[-1] > self.in_channels:
                tile = tile[:, :, : self.in_channels]
            else:
                # Pad with zeros
                pad_channels = self.in_channels - tile.shape[-1]
                tile = np.concatenate(
                    [tile, np.zeros((*tile.shape[:2], pad_channels), dtype=tile.dtype)],
                    axis=-1,
                )

        # Normalize to [0, 1]
        if tile.max() > 1.0:
            tile = tile.astype(np.float32) / 255.0
        else:
            tile = tile.astype(np.float32)

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            probs = self.model(x)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert to numpy
        probs_np = probs.squeeze().cpu().numpy()
        mask = (probs_np > self.threshold).astype(np.uint8)
        cloud_fraction = float(mask.mean())

        return {
            "mask": mask,
            "probabilities": probs_np,
            "cloud_fraction": cloud_fraction,
            "inference_time_ms": inference_time_ms,
        }

    def get_compute_cost(self) -> float:
        """Get compute cost in TOPS."""
        return self._COMPUTE_COST_TOPS

    def get_memory_footprint(self) -> float:
        """Get memory footprint in MB.

        Lightweight U-Net: ~5 MB weights.
        """
        return 5.0

    def save_weights(self, path: str | Path) -> Path:
        """Save model weights.

        Args:
            path: Output path for weights.

        Returns:
            Path to saved weights.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    def export_onnx(
        self, path: str | Path, input_shape: tuple = (1, 4, 384, 384)
    ) -> Path:
        """Export model to ONNX format.

        Args:
            path: Output path for ONNX model.
            input_shape: Input tensor shape (B, C, H, W).

        Returns:
            Path to exported model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=11,
        )

        return path
