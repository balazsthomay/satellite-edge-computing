"""Tests for CV models."""

import numpy as np
import pytest
from pathlib import Path

from satellite_edge.data import create_synthetic_tile
from satellite_edge.models import CloudMaskModel, ImageCompressor, SatelliteDetector


class TestSatelliteDetector:
    """Tests for YOLOv8n-based object detector."""

    @pytest.fixture
    def detector(self):
        return SatelliteDetector(device="cpu", conf_threshold=0.25)

    def test_init(self, detector):
        assert detector.name == "SatelliteDetector"
        assert detector.task_type == 0  # DETECTION

    def test_predict_returns_expected_keys(self, detector):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = detector.predict(tile)

        assert "boxes" in result
        assert "scores" in result
        assert "classes" in result
        assert "class_names" in result
        assert "n_detections" in result
        assert "inference_time_ms" in result

    def test_predict_handles_rgb(self, detector):
        tile = create_synthetic_tile(384, 384, 3, "urban", seed=42)
        result = detector.predict(tile)
        assert "n_detections" in result

    def test_predict_handles_rgbnir(self, detector):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = detector.predict(tile)
        assert "n_detections" in result

    def test_get_compute_cost(self, detector):
        cost = detector.get_compute_cost()
        assert cost > 0
        assert cost < 0.1  # YOLOv8n is lightweight

    def test_get_memory_footprint(self, detector):
        mem = detector.get_memory_footprint()
        assert mem > 0
        assert mem < 20  # Should be small

    def test_filter_satellite_classes_empty(self, detector):
        predictions = {
            "boxes": np.array([]).reshape(0, 4),
            "scores": np.array([]),
            "classes": np.array([], dtype=int),
            "class_names": [],
            "n_detections": 0,
            "inference_time_ms": 10.0,
        }
        filtered = detector.filter_satellite_classes(predictions)
        assert filtered["n_detections"] == 0

    def test_inference_time_measured(self, detector):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = detector.predict(tile)
        assert result["inference_time_ms"] > 0


class TestCloudMaskModel:
    """Tests for cloud mask segmentation model."""

    @pytest.fixture
    def model(self):
        return CloudMaskModel(in_channels=4, device="cpu")

    def test_init(self, model):
        assert model.name == "CloudMaskModel"
        assert model.task_type == 2  # CLOUD_MASK

    def test_predict_returns_expected_keys(self, model):
        tile = create_synthetic_tile(384, 384, 4, "cloud", seed=42)
        result = model.predict(tile)

        assert "mask" in result
        assert "probabilities" in result
        assert "cloud_fraction" in result
        assert "inference_time_ms" in result

    def test_predict_mask_shape(self, model):
        tile = create_synthetic_tile(384, 384, 4, "cloud", seed=42)
        result = model.predict(tile)

        assert result["mask"].shape == (384, 384)
        assert result["mask"].dtype == np.uint8

    def test_predict_probabilities_range(self, model):
        tile = create_synthetic_tile(384, 384, 4, "cloud", seed=42)
        result = model.predict(tile)

        assert result["probabilities"].min() >= 0.0
        assert result["probabilities"].max() <= 1.0

    def test_predict_cloud_fraction_range(self, model):
        tile = create_synthetic_tile(384, 384, 4, "cloud", seed=42)
        result = model.predict(tile)

        assert 0.0 <= result["cloud_fraction"] <= 1.0

    def test_predict_handles_rgb(self, model):
        # RGB input with 4-channel model (should pad)
        tile = create_synthetic_tile(384, 384, 3, "cloud", seed=42)
        result = model.predict(tile)
        assert "mask" in result

    def test_predict_handles_5_channel(self):
        # 5-channel input with 4-channel model (should truncate)
        model = CloudMaskModel(in_channels=4, device="cpu")
        tile = np.zeros((384, 384, 5), dtype=np.uint8)
        result = model.predict(tile)
        assert "mask" in result

    def test_get_compute_cost(self, model):
        cost = model.get_compute_cost()
        assert cost > 0
        assert cost < 0.01  # Lightweight U-Net

    def test_get_memory_footprint(self, model):
        mem = model.get_memory_footprint()
        assert mem > 0
        assert mem < 20

    def test_save_weights(self, model, tmp_path):
        path = model.save_weights(tmp_path / "cloud_mask.pt")
        assert path.exists()

    @pytest.mark.skip(reason="Requires onnxscript which is optional")
    def test_export_onnx(self, model, tmp_path):
        path = model.export_onnx(tmp_path / "cloud_mask.onnx")
        assert path.exists()


class TestImageCompressor:
    """Tests for convolutional autoencoder compressor."""

    @pytest.fixture
    def model(self):
        return ImageCompressor(in_channels=4, latent_dim=64, device="cpu")

    def test_init(self, model):
        assert model.name == "ImageCompressor"
        assert model.task_type == 3  # COMPRESSION

    def test_predict_returns_expected_keys(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = model.predict(tile)

        assert "latent" in result
        assert "reconstructed" in result
        assert "compression_ratio" in result
        assert "mse" in result
        assert "ssim" in result
        assert "inference_time_ms" in result

    def test_predict_latent_shape(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = model.predict(tile)

        # 384 -> 48 (8x reduction)
        assert result["latent"].shape == (64, 48, 48)

    def test_predict_reconstructed_shape(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = model.predict(tile)

        assert result["reconstructed"].shape == (384, 384, 4)
        assert result["reconstructed"].dtype == np.uint8

    def test_predict_compression_ratio(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = model.predict(tile)

        # Expected: (384*384*4) / (48*48*64) = 4.0
        assert abs(result["compression_ratio"] - 4.0) < 0.1

    def test_predict_metrics_reasonable(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)
        result = model.predict(tile)

        # MSE should be positive
        assert result["mse"] >= 0

        # SSIM in [-1, 1] but usually positive
        assert -1 <= result["ssim"] <= 1

    def test_encode_decode_roundtrip(self, model):
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)

        latent = model.encode(tile)
        reconstructed = model.decode(latent)

        # Shapes should match
        assert reconstructed.shape == tile.shape[:2] + (4,)

    def test_get_compute_cost(self, model):
        cost = model.get_compute_cost()
        assert cost > 0
        assert cost < 0.01  # Lightweight CAE

    def test_get_memory_footprint(self, model):
        mem = model.get_memory_footprint()
        assert mem > 0
        assert mem < 10

    def test_compression_ratio_method(self, model):
        ratio = model.compression_ratio()
        # 8x8 spatial reduction, 4 in -> 64 out channels
        assert ratio > 0

    def test_save_weights(self, model, tmp_path):
        path = model.save_weights(tmp_path / "compressor.pt")
        assert path.exists()

    @pytest.mark.skip(reason="Requires onnxscript which is optional")
    def test_export_onnx(self, model, tmp_path):
        path = model.export_onnx(tmp_path / "compressor.onnx")
        assert path.exists()


class TestModelInferenceTiming:
    """Tests for inference timing requirements."""

    @pytest.mark.slow
    def test_cloud_mask_inference_time(self):
        model = CloudMaskModel(in_channels=4, device="cpu")
        tile = create_synthetic_tile(384, 384, 4, "cloud", seed=42)

        # Warmup
        model.predict(tile)

        # Measure
        result = model.predict(tile)

        # Should be under 200ms on CPU
        assert result["inference_time_ms"] < 200

    @pytest.mark.slow
    def test_compressor_inference_time(self):
        model = ImageCompressor(in_channels=4, device="cpu")
        tile = create_synthetic_tile(384, 384, 4, "urban", seed=42)

        # Warmup
        model.predict(tile)

        # Measure
        result = model.predict(tile)

        # Should be under 100ms on CPU
        assert result["inference_time_ms"] < 100
