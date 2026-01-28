"""Image compression using convolutional autoencoder.

Lightweight autoencoder for onboard satellite image compression.
Designed for edge deployment with minimal compute requirements.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from satellite_edge.models.base import BaseCVModel


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for image compression.

    Compresses images to a latent space with configurable compression ratio.
    Uses strided convolutions for downsampling and transposed convolutions
    for upsampling.
    """

    def __init__(self, in_channels: int = 4, latent_dim: int = 64):
        """Initialize autoencoder.

        Args:
            in_channels: Number of input channels.
            latent_dim: Number of channels in latent space.
        """
        super().__init__()

        # Encoder: 384x384 -> 48x48 (8x spatial reduction)
        self.encoder = nn.Sequential(
            # 384 -> 192
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 192 -> 96
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 96 -> 48
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder: 48x48 -> 384x384
        self.decoder = nn.Sequential(
            # 48 -> 96
            nn.ConvTranspose2d(
                latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 96 -> 192
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 192 -> 384
            nn.ConvTranspose2d(
                32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.latent_dim = latent_dim
        self.in_channels = in_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns both reconstruction and latent."""
        z = self.encode(x)
        return self.decode(z), z


class ImageCompressor(BaseCVModel):
    """Image compressor for satellite imagery.

    Uses a convolutional autoencoder to compress satellite tiles.
    Achieves ~64x compression (384x384x4 -> 48x48x64 latent).

    Attributes:
        model: Underlying autoencoder.
        device: Device for inference.
    """

    name = "ImageCompressor"
    task_type = 3  # COMPRESSION

    # CAE: ~0.5 GFLOPs = 0.0005 TOPS per inference
    _COMPUTE_COST_TOPS = 0.0005

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 64,
        weights_path: str | Path | None = None,
        device: str = "cpu",
    ):
        """Initialize compressor.

        Args:
            in_channels: Number of input channels.
            latent_dim: Number of channels in latent space.
            weights_path: Path to pretrained weights. If None, uses random init.
            device: Device for inference.
        """
        self.device = device
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.model = ConvAutoencoder(in_channels=in_channels, latent_dim=latent_dim)

        if weights_path is not None:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=device, weights_only=True)
            )

        self.model.to(device)
        self.model.eval()

    def predict(self, tile: np.ndarray) -> dict[str, Any]:
        """Compress and reconstruct a tile.

        Args:
            tile: Array of shape (H, W, C) with pixel values.

        Returns:
            Dict with:
                - 'latent': Compressed latent array
                - 'reconstructed': Reconstructed tile
                - 'compression_ratio': Achieved compression ratio
                - 'mse': Mean squared error
                - 'ssim': Structural similarity (approximate)
                - 'inference_time_ms': Time taken for inference
        """
        # Handle channel mismatch
        if tile.shape[-1] != self.in_channels:
            if tile.shape[-1] > self.in_channels:
                tile = tile[:, :, : self.in_channels]
            else:
                pad_channels = self.in_channels - tile.shape[-1]
                tile = np.concatenate(
                    [tile, np.zeros((*tile.shape[:2], pad_channels), dtype=tile.dtype)],
                    axis=-1,
                )

        # Normalize to [0, 1]
        if tile.max() > 1.0:
            tile_norm = tile.astype(np.float32) / 255.0
        else:
            tile_norm = tile.astype(np.float32)

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        x = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            reconstructed, latent = self.model(x)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert outputs
        latent_np = latent.squeeze().cpu().numpy()
        recon_np = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()

        # Compute metrics
        mse = float(np.mean((tile_norm - recon_np) ** 2))
        ssim = self._approximate_ssim(tile_norm, recon_np)

        # Compression ratio
        # Original: H * W * C, Latent: H/8 * W/8 * latent_dim
        original_size = np.prod(tile.shape)
        latent_size = np.prod(latent_np.shape)
        compression_ratio = original_size / latent_size

        return {
            "latent": latent_np,
            "reconstructed": (recon_np * 255).astype(np.uint8),
            "compression_ratio": compression_ratio,
            "mse": mse,
            "ssim": ssim,
            "inference_time_ms": inference_time_ms,
        }

    def encode(self, tile: np.ndarray) -> np.ndarray:
        """Encode tile to latent representation.

        Args:
            tile: Array of shape (H, W, C).

        Returns:
            Latent array of shape (H/8, W/8, latent_dim).
        """
        result = self.predict(tile)
        return result["latent"]

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent representation to image.

        Args:
            latent: Latent array of shape (latent_dim, H, W) or (H, W, latent_dim).

        Returns:
            Reconstructed array of shape (H*8, W*8, in_channels).
        """
        # Handle different input formats
        if latent.shape[-1] == self.latent_dim:
            # (H, W, C) format -> (C, H, W)
            latent = np.transpose(latent, (2, 0, 1))

        # Add batch dimension
        z = torch.from_numpy(latent).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            reconstructed = self.model.decode(z)

        recon_np = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()
        return (recon_np * 255).astype(np.uint8)

    def _approximate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Approximate structural similarity.

        Simplified SSIM without full windowed calculation.
        """
        c1 = 0.01**2
        c2 = 0.03**2

        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

        num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

        return float(num / den)

    def get_compute_cost(self) -> float:
        """Get compute cost in TOPS."""
        return self._COMPUTE_COST_TOPS

    def get_memory_footprint(self) -> float:
        """Get memory footprint in MB.

        CAE: ~2 MB weights.
        """
        return 2.0

    def compression_ratio(self) -> float:
        """Get theoretical compression ratio.

        For 384x384x4 input with 64-channel 48x48 latent:
        (384*384*4) / (48*48*64) = 4
        """
        spatial_reduction = 8  # 384/48
        channel_change = self.in_channels / self.latent_dim
        return (spatial_reduction**2) * (1 / channel_change)

    def save_weights(self, path: str | Path) -> Path:
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return path

    def export_onnx(
        self, path: str | Path, input_shape: tuple = (1, 4, 384, 384)
    ) -> Path:
        """Export model to ONNX format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=["input"],
            output_names=["reconstructed", "latent"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "reconstructed": {0: "batch", 2: "height", 3: "width"},
                "latent": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=11,
        )

        return path
