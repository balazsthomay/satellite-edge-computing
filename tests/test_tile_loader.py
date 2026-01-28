"""Tests for tile loader and data pipeline."""

import numpy as np
import pytest
from pathlib import Path

from satellite_edge.data import TileLoader, create_synthetic_tile, save_synthetic_tile


class TestTileLoader:
    """Tests for TileLoader class."""

    def test_init_default_tile_size(self):
        loader = TileLoader()
        assert loader.tile_size == 384

    def test_init_custom_tile_size(self):
        loader = TileLoader(tile_size=256)
        assert loader.tile_size == 256

    def test_init_invalid_tile_size(self):
        with pytest.raises(ValueError):
            TileLoader(tile_size=0)
        with pytest.raises(ValueError):
            TileLoader(tile_size=-1)

    def test_load_tile_file_not_found(self):
        loader = TileLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_tile("/nonexistent/path.tif")

    def test_load_tile_from_geotiff(self, tmp_path):
        # Create a test GeoTIFF
        tile_path = save_synthetic_tile(
            tmp_path / "test.tif", tile_type="urban", height=128, width=128, seed=42
        )

        loader = TileLoader()
        tile = loader.load_tile(tile_path)

        assert tile.shape == (128, 128, 4)
        assert tile.dtype == np.uint8

    def test_generate_tiles(self, tmp_path):
        # Create a larger image
        tile_path = save_synthetic_tile(
            tmp_path / "large.tif",
            tile_type="mixed",
            height=256,
            width=256,
            seed=42,
        )

        loader = TileLoader(tile_size=128)
        tiles = list(loader.generate_tiles(tile_path))

        # 256x256 with 128x128 tiles = 4 tiles
        assert len(tiles) == 4

        for tile, window, metadata in tiles:
            assert tile.shape == (128, 128, 4)
            assert "row" in metadata
            assert "col" in metadata
            assert "bounds" in metadata

    def test_get_tile_count(self, tmp_path):
        tile_path = save_synthetic_tile(
            tmp_path / "test.tif", height=384, width=384, seed=42
        )

        loader = TileLoader(tile_size=128)
        count = loader.get_tile_count(tile_path)
        assert count == 9  # 3x3 tiles

    def test_load_from_array_hwc_format(self):
        loader = TileLoader()
        # Already (H, W, C) format
        array = np.zeros((100, 100, 4), dtype=np.uint8)
        result = loader.load_from_array(array)
        assert result.shape == (100, 100, 4)

    def test_load_from_array_chw_format(self):
        loader = TileLoader()
        # (C, H, W) format - should transpose
        array = np.zeros((4, 100, 100), dtype=np.uint8)
        result = loader.load_from_array(array)
        assert result.shape == (100, 100, 4)

    def test_load_from_array_invalid_dims(self):
        loader = TileLoader()
        array = np.zeros((100, 100), dtype=np.uint8)  # 2D
        with pytest.raises(ValueError):
            loader.load_from_array(array)


class TestSyntheticTiles:
    """Tests for synthetic tile generation."""

    def test_create_urban_tile(self):
        tile = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        assert tile.shape == (128, 128, 4)
        assert tile.dtype == np.uint8

    def test_create_water_tile(self):
        tile = create_synthetic_tile(128, 128, 4, "water", seed=42)
        assert tile.shape == (128, 128, 4)
        # Water should be dark (low values)
        assert tile[:, :, :3].mean() < 100

    def test_create_forest_tile(self):
        tile = create_synthetic_tile(128, 128, 4, "forest", seed=42)
        assert tile.shape == (128, 128, 4)
        # Forest should have high NIR
        assert tile[:, :, 3].mean() > 100

    def test_create_cloud_tile(self):
        tile = create_synthetic_tile(128, 128, 4, "cloud", seed=42)
        assert tile.shape == (128, 128, 4)
        # Clouds should be bright
        assert tile.mean() > 150

    def test_create_mixed_tile(self):
        tile = create_synthetic_tile(256, 256, 4, "mixed", seed=42)
        assert tile.shape == (256, 256, 4)

    def test_create_rgb_tile(self):
        tile = create_synthetic_tile(128, 128, 3, "urban", seed=42)
        assert tile.shape == (128, 128, 3)

    def test_reproducibility(self):
        tile1 = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        tile2 = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        np.testing.assert_array_equal(tile1, tile2)

    def test_different_seeds_differ(self):
        tile1 = create_synthetic_tile(128, 128, 4, "urban", seed=42)
        tile2 = create_synthetic_tile(128, 128, 4, "urban", seed=99)
        assert not np.array_equal(tile1, tile2)

    def test_save_synthetic_tile(self, tmp_path):
        path = save_synthetic_tile(
            tmp_path / "test.tif", tile_type="urban", height=64, width=64, seed=42
        )
        assert path.exists()
        assert path.suffix == ".tif"

        # Verify it's a valid GeoTIFF
        loader = TileLoader()
        tile = loader.load_tile(path)
        assert tile.shape == (64, 64, 4)
