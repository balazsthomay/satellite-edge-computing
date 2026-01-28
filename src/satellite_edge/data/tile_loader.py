"""Tile loader for satellite imagery.

Handles loading GeoTIFF tiles and generating tile windows from larger images.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


class TileLoader:
    """Load and tile satellite imagery from GeoTIFF files.

    Supports loading individual tiles or generating non-overlapping tiles
    from larger satellite images.

    Attributes:
        tile_size: Size of tiles to generate (square tiles).
    """

    def __init__(self, tile_size: int = 384):
        """Initialize tile loader.

        Args:
            tile_size: Width/height of tiles to generate. Default 384 matches
                common satellite imagery processing pipelines.
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        self.tile_size = tile_size

    def load_tile(self, path: str | Path, window: Window | None = None) -> np.ndarray:
        """Load a tile from a GeoTIFF file.

        Args:
            path: Path to GeoTIFF file.
            window: Optional rasterio Window for reading a subset.

        Returns:
            Array of shape (H, W, C) with pixel values.

        Raises:
            FileNotFoundError: If file doesn't exist.
            rasterio.errors.RasterioIOError: If file can't be read.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tile not found: {path}")

        with rasterio.open(path) as src:
            if window is None:
                data = src.read()  # (C, H, W)
            else:
                data = src.read(window=window)

            # Transpose to (H, W, C) for CV models
            return np.transpose(data, (1, 2, 0))

    def generate_tiles(
        self, path: str | Path
    ) -> Iterator[tuple[np.ndarray, Window, dict]]:
        """Generate non-overlapping tiles from a large image.

        Args:
            path: Path to GeoTIFF file.

        Yields:
            Tuples of (tile_array, window, metadata) where:
                - tile_array: Array of shape (H, W, C)
                - window: rasterio Window defining tile location
                - metadata: Dict with 'row', 'col', 'bounds' keys
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with rasterio.open(path) as src:
            for row in range(0, src.height, self.tile_size):
                for col in range(0, src.width, self.tile_size):
                    # Calculate actual window size (may be smaller at edges)
                    win_width = min(self.tile_size, src.width - col)
                    win_height = min(self.tile_size, src.height - row)

                    window = Window(col, row, win_width, win_height)
                    tile = src.read(window=window)

                    # Get geographic bounds for this tile
                    bounds = rasterio.windows.bounds(window, src.transform)

                    metadata = {
                        "row": row // self.tile_size,
                        "col": col // self.tile_size,
                        "bounds": bounds,
                        "crs": str(src.crs) if src.crs else None,
                    }

                    yield np.transpose(tile, (1, 2, 0)), window, metadata

    def get_tile_count(self, path: str | Path) -> int:
        """Get the number of tiles that would be generated from an image.

        Args:
            path: Path to GeoTIFF file.

        Returns:
            Number of tiles.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with rasterio.open(path) as src:
            n_cols = (src.width + self.tile_size - 1) // self.tile_size
            n_rows = (src.height + self.tile_size - 1) // self.tile_size
            return n_cols * n_rows

    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Normalize an array to expected format.

        Args:
            array: Input array, can be (C, H, W) or (H, W, C).

        Returns:
            Array of shape (H, W, C).
        """
        if array.ndim != 3:
            raise ValueError(f"Expected 3D array, got {array.ndim}D")

        # Heuristic: if first dim is small (1-4), assume (C, H, W)
        if array.shape[0] <= 4 and array.shape[0] < array.shape[1]:
            return np.transpose(array, (1, 2, 0))
        return array
