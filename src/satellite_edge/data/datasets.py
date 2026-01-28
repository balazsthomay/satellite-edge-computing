"""Dataset utilities for satellite imagery.

Provides helpers for accessing Sentinel-2 via STAC and generating
synthetic test tiles.
"""

from pathlib import Path

import numpy as np


def create_synthetic_tile(
    height: int = 384,
    width: int = 384,
    channels: int = 4,
    tile_type: str = "urban",
    seed: int | None = None,
) -> np.ndarray:
    """Create a synthetic satellite tile for testing.

    Generates tiles with characteristics matching different land cover types.
    Useful for testing without downloading real imagery.

    Args:
        height: Tile height in pixels.
        width: Tile width in pixels.
        channels: Number of channels (3=RGB, 4=RGBNIR).
        tile_type: One of 'urban', 'water', 'forest', 'cloud', 'mixed'.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (H, W, C) with uint8 values [0, 255].
    """
    rng = np.random.default_rng(seed)
    tile = np.zeros((height, width, channels), dtype=np.uint8)

    if tile_type == "urban":
        # High contrast, many edges (buildings, roads)
        base = rng.integers(60, 120, size=(height, width), dtype=np.uint8)
        # Add grid pattern (streets)
        for i in range(0, height, 50):
            base[i : i + 3, :] = rng.integers(40, 80)
        for j in range(0, width, 50):
            base[:, j : j + 3] = rng.integers(40, 80)
        # Add building-like rectangles
        for _ in range(30):
            h, w = rng.integers(10, 40, size=2)
            y, x = rng.integers(0, height - h), rng.integers(0, width - w)
            brightness = rng.integers(100, 200)
            base[y : y + h, x : x + w] = brightness
        for c in range(min(3, channels)):
            tile[:, :, c] = base + rng.integers(-20, 20, size=(height, width))
        if channels > 3:
            # NIR: vegetation appears bright
            tile[:, :, 3] = base + 40

    elif tile_type == "water":
        # Low values, low variance (dark, uniform)
        for c in range(channels):
            if c < 3:
                tile[:, :, c] = rng.integers(
                    20, 50, size=(height, width), dtype=np.uint8
                )
            else:
                # NIR: water is very dark
                tile[:, :, c] = rng.integers(
                    5, 20, size=(height, width), dtype=np.uint8
                )
        # Add some wave texture
        for i in range(height):
            wave = int(5 * np.sin(i / 10))
            tile[i, :, 0] = np.clip(
                tile[i, :, 0].astype(np.int16) + wave, 0, 255
            ).astype(np.uint8)

    elif tile_type == "forest":
        # Green-dominant, medium variance
        tile[:, :, 0] = rng.integers(30, 70, size=(height, width), dtype=np.uint8)  # R
        tile[:, :, 1] = rng.integers(60, 120, size=(height, width), dtype=np.uint8)  # G
        tile[:, :, 2] = rng.integers(20, 60, size=(height, width), dtype=np.uint8)  # B
        if channels > 3:
            # NIR: vegetation is very bright
            tile[:, :, 3] = rng.integers(150, 220, size=(height, width), dtype=np.uint8)
        # Add some texture variation
        noise = rng.integers(-15, 15, size=(height, width))
        for c in range(channels):
            tile[:, :, c] = np.clip(tile[:, :, c].astype(int) + noise, 0, 255).astype(
                np.uint8
            )

    elif tile_type == "cloud":
        # Bright, uniform-ish (simulates cloud cover)
        base = rng.integers(200, 240, size=(height, width), dtype=np.uint8)
        # Add cloud-like blobs
        for _ in range(5):
            cy, cx = rng.integers(0, height), rng.integers(0, width)
            radius = rng.integers(30, 80)
            for y in range(max(0, cy - radius), min(height, cy + radius)):
                for x in range(max(0, cx - radius), min(width, cx + radius)):
                    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                    if dist < radius:
                        brightness = int(255 - 30 * dist / radius)
                        base[y, x] = max(base[y, x], brightness)
        for c in range(channels):
            tile[:, :, c] = base

    else:  # mixed
        # Combination of features
        # Top-left: urban
        tile[: height // 2, : width // 2] = create_synthetic_tile(
            height // 2, width // 2, channels, "urban", seed
        )
        # Top-right: forest
        tile[: height // 2, width // 2 :] = create_synthetic_tile(
            height // 2, width - width // 2, channels, "forest", seed
        )
        # Bottom-left: water
        tile[height // 2 :, : width // 2] = create_synthetic_tile(
            height - height // 2, width // 2, channels, "water", seed
        )
        # Bottom-right: cloud
        tile[height // 2 :, width // 2 :] = create_synthetic_tile(
            height - height // 2, width - width // 2, channels, "cloud", seed
        )

    return np.clip(tile, 0, 255).astype(np.uint8)


def save_synthetic_tile(
    path: str | Path,
    tile_type: str = "urban",
    height: int = 384,
    width: int = 384,
    channels: int = 4,
    seed: int | None = None,
) -> Path:
    """Save a synthetic tile as a GeoTIFF.

    Args:
        path: Output path for the GeoTIFF.
        tile_type: Type of synthetic tile to generate.
        height: Tile height.
        width: Tile width.
        channels: Number of channels.
        seed: Random seed.

    Returns:
        Path to saved file.
    """
    import rasterio
    from rasterio.transform import from_bounds

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tile = create_synthetic_tile(height, width, channels, tile_type, seed)

    # Create a simple transform (arbitrary location)
    transform = from_bounds(-122.5, 37.5, -122.0, 38.0, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=tile.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        # rasterio expects (C, H, W)
        dst.write(np.transpose(tile, (2, 0, 1)))

    return path


def generate_sample_dataset(
    output_dir: str | Path,
    n_tiles: int = 10,
    seed: int = 42,
) -> list[Path]:
    """Generate a sample dataset of synthetic tiles.

    Creates tiles of various types for testing the full pipeline.

    Args:
        output_dir: Directory to save tiles.
        n_tiles: Number of tiles to generate.
        seed: Random seed.

    Returns:
        List of paths to generated tiles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_types = ["urban", "water", "forest", "cloud", "mixed"]
    paths = []

    for i in range(n_tiles):
        tile_type = tile_types[i % len(tile_types)]
        path = output_dir / f"tile_{i:03d}_{tile_type}.tif"
        save_synthetic_tile(path, tile_type=tile_type, seed=seed + i)
        paths.append(path)

    return paths
