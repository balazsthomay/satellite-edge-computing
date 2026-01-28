#!/usr/bin/env python3
"""Phase 3 Demo: Real imagery through RL scheduler.

Demonstrates the end-to-end pipeline:
1. Generate/load satellite imagery tiles
2. Estimate tile values based on content (cloud, entropy, edges)
3. Create ImageTasks with content-aware value
4. Run scheduling with PPO agent or baselines
5. Execute CV models on scheduled tasks
6. Display results

Usage:
    uv run scripts/demo_phase3.py [--tiles N] [--steps N] [--output DIR]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from satellite_edge.data import (
    TileLoader,
    ValueEstimator,
    create_synthetic_tile,
    generate_sample_dataset,
)
from satellite_edge.environment import ImageTask, SatelliteEnv, TaskType
from satellite_edge.environment.satellite_env import EpisodeConfig, SatelliteConfig
from satellite_edge.models import CloudMaskModel, ImageCompressor, SatelliteDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Demo: Real imagery pipeline")
    parser.add_argument(
        "--tiles", type=int, default=5, help="Number of tiles to process"
    )
    parser.add_argument("--steps", type=int, default=50, help="Episode steps to run")
    parser.add_argument(
        "--output", type=str, default="results/phase3_demo", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-detection", action="store_true", help="Skip YOLO detection (faster)"
    )
    return parser.parse_args()


def load_cv_models(skip_detection: bool = False) -> dict:
    """Load CV models for each task type."""
    models = {}

    if not skip_detection:
        print("Loading YOLOv8n detector...")
        models[TaskType.DETECTION] = SatelliteDetector(device="cpu")

    print("Loading cloud mask model...")
    models[TaskType.CLOUD_MASK] = CloudMaskModel(in_channels=4, device="cpu")

    print("Loading image compressor...")
    models[TaskType.COMPRESSION] = ImageCompressor(in_channels=4, device="cpu")

    return models


def run_cv_inference(
    model,
    tile: np.ndarray,
    task_type: TaskType,
) -> dict:
    """Run CV model inference on a tile."""
    result = model.predict(tile)

    # Add task-specific summary
    if task_type == TaskType.DETECTION:
        result["summary"] = f"{result['n_detections']} objects detected"
    elif task_type == TaskType.CLOUD_MASK:
        result["summary"] = f"{result['cloud_fraction'] * 100:.1f}% cloud coverage"
    elif task_type == TaskType.COMPRESSION:
        result["summary"] = (
            f"{result['compression_ratio']:.1f}x compression, SSIM={result['ssim']:.3f}"
        )
    else:
        result["summary"] = "Processed"

    return result


def create_visualization(
    tiles: list[tuple[np.ndarray, dict]],
    results: list[dict],
    output_dir: Path,
) -> Path:
    """Create visualization of tiles, metrics, and CV outputs."""
    n_tiles = len(tiles)
    fig, axes = plt.subplots(n_tiles, 4, figsize=(16, 4 * n_tiles))

    if n_tiles == 1:
        axes = axes.reshape(1, -1)

    for i, ((tile, metrics), result) in enumerate(zip(tiles, results)):
        # Original tile (RGB only)
        ax = axes[i, 0]
        rgb = tile[:, :, :3] if tile.shape[-1] >= 3 else tile
        if rgb.max() > 1:
            rgb = rgb.astype(np.float32) / 255.0
        ax.imshow(rgb)
        ax.set_title(f"Tile {i}: {metrics.get('tile_type', 'unknown')}")
        ax.axis("off")

        # Value metrics
        ax = axes[i, 1]
        metrics_text = (
            f"Cloud: {metrics['cloud_fraction'] * 100:.1f}%\n"
            f"Entropy: {metrics['entropy']:.2f}\n"
            f"Edge density: {metrics['edge_density']:.2f}\n"
            f"Task type: {result.get('task_type', 'N/A')}\n"
            f"Value: {result.get('value', 0):.2f}"
        )
        ax.text(
            0.1,
            0.5,
            metrics_text,
            fontsize=12,
            family="monospace",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_title("Value Metrics")
        ax.axis("off")

        # CV output visualization
        ax = axes[i, 2]
        if "mask" in result:
            ax.imshow(result["mask"], cmap="gray")
            ax.set_title(f"Cloud Mask ({result['cloud_fraction'] * 100:.1f}%)")
        elif "reconstructed" in result:
            recon = result["reconstructed"]
            if recon.shape[-1] >= 3:
                recon = recon[:, :, :3]
            if recon.max() > 1:
                recon = recon.astype(np.float32) / 255.0
            ax.imshow(recon)
            ax.set_title(f"Reconstructed ({result['compression_ratio']:.1f}x)")
        elif "boxes" in result and len(result["boxes"]) > 0:
            ax.imshow(rgb)
            for box, cls_name in zip(result["boxes"], result["class_names"]):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, cls_name, color="red", fontsize=8)
            ax.set_title(f"Detections ({result['n_detections']})")
        else:
            ax.imshow(rgb)
            ax.set_title("No CV output")
        ax.axis("off")

        # Timing info
        ax = axes[i, 3]
        timing_text = (
            f"Inference: {result.get('inference_time_ms', 0):.1f}ms\n"
            f"Summary: {result.get('summary', 'N/A')}"
        )
        ax.text(
            0.1,
            0.5,
            timing_text,
            fontsize=12,
            family="monospace",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_title("Performance")
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "demo_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print("=" * 60)
    print("PHASE 3 DEMO: Real Imagery Pipeline")
    print("=" * 60)
    print()

    # 1. Generate synthetic tiles
    print("1. Generating synthetic satellite tiles...")
    tile_types = ["urban", "water", "forest", "cloud", "mixed"]
    value_estimator = ValueEstimator()

    tiles_with_metrics = []
    for i in range(args.tiles):
        tile_type = tile_types[i % len(tile_types)]
        tile = create_synthetic_tile(384, 384, 4, tile_type, seed=args.seed + i)
        metrics = value_estimator.compute_all_metrics(tile)
        metrics["tile_type"] = tile_type
        tiles_with_metrics.append((tile, metrics))
        print(
            f"   Tile {i} ({tile_type}): cloud={metrics['cloud_fraction']:.2f}, "
            f"entropy={metrics['entropy']:.2f}, edges={metrics['edge_density']:.2f}"
        )
    print()

    # 2. Load CV models
    print("2. Loading CV models...")
    models = load_cv_models(skip_detection=args.no_detection)
    print()

    # 3. Create ImageTasks and assign task types
    print("3. Creating ImageTasks with content-aware value...")
    task_assignments = [
        TaskType.DETECTION if not args.no_detection else TaskType.CLOUD_MASK,
        TaskType.CLOUD_MASK,
        TaskType.COMPRESSION,
        TaskType.CLOUD_MASK,
        TaskType.COMPRESSION,
    ]

    tasks = []
    for i, ((tile, metrics), task_type) in enumerate(
        zip(tiles_with_metrics, task_assignments[: len(tiles_with_metrics)])
    ):
        task = ImageTask(
            task_type=task_type,
            tile_id=i,
            arrival_time=0.0,
            cloud_fraction=metrics["cloud_fraction"],
            entropy_score=metrics["entropy"],
            edge_density=metrics["edge_density"],
        )
        value = task.current_value(0.0)
        quality = task.image_quality_multiplier
        tasks.append((task, tile))
        print(
            f"   Task {i}: {TaskType(task_type).name}, "
            f"quality_mult={quality:.2f}, value={value:.2f}"
        )
    print()

    # 4. Run CV inference on each task
    print("4. Running CV inference...")
    results = []
    total_inference_time = 0

    for i, (task, tile) in enumerate(tasks):
        task_type = task.task_type

        if task_type not in models:
            # Skip tasks without loaded models
            result = {
                "inference_time_ms": 0,
                "summary": f"Skipped (no model for {TaskType(task_type).name})",
                "task_type": TaskType(task_type).name,
                "value": task.current_value(0.0),
            }
        else:
            model = models[task_type]
            result = run_cv_inference(model, tile, task_type)
            result["task_type"] = TaskType(task_type).name
            result["value"] = task.current_value(0.0)
            total_inference_time += result["inference_time_ms"]

        results.append(result)
        print(
            f"   Task {i} ({result['task_type']}): {result['summary']} "
            f"[{result['inference_time_ms']:.1f}ms]"
        )

    print(f"\n   Total inference time: {total_inference_time:.1f}ms")
    print()

    # 5. Create visualization
    print("5. Creating visualization...")
    viz_path = create_visualization(tiles_with_metrics, results, output_dir)
    print(f"   Saved to: {viz_path}")
    print()

    # 6. Summary statistics
    print("6. Summary")
    print("-" * 40)
    total_value = sum(r.get("value", 0) for r in results)
    avg_inference = total_inference_time / len(results) if results else 0
    print(f"   Tiles processed: {len(tiles_with_metrics)}")
    print(f"   Total value captured: {total_value:.2f}")
    print(f"   Avg inference time: {avg_inference:.1f}ms per tile")
    print(f"   Models loaded: {list(models.keys())}")
    print()

    # 7. Demonstrate value estimation differences
    print("7. Value Estimation by Task Type")
    print("-" * 40)
    sample_tile, sample_metrics = tiles_with_metrics[0]
    for task_type in TaskType:
        task = ImageTask(
            task_type=task_type,
            tile_id=0,
            arrival_time=0.0,
            cloud_fraction=sample_metrics["cloud_fraction"],
            entropy_score=sample_metrics["entropy"],
            edge_density=sample_metrics["edge_density"],
        )
        print(
            f"   {TaskType(task_type).name:12s}: "
            f"quality_mult={task.image_quality_multiplier:.2f}, "
            f"value={task.current_value(0.0):.2f}"
        )
    print()

    print("=" * 60)
    print("Demo complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
