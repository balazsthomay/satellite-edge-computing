# Satellite Edge Computing Resource Optimizer

RL-based onboard compute scheduler for Earth observation satellites. Unlike task offloading research (ground ↔ satellite), this system optimizes **what to compute** given fixed, limited onboard resources.

## Problem

Earth observation satellites face a fundamental mismatch:

- **Data volume**: ~100 TB/day per satellite, 230 PB/day fleet-wide by 2032
- **Compute gap**: Space-grade chips trail ground hardware by 3-4 orders of magnitude (radiation hardening)
- **Limited windows**: Only ~15% of orbit has ground station visibility for downlink

Companies like Satellogic are embedding AI chips directly on satellites, but there's no intelligent system to dynamically allocate that scarce compute across competing CV tasks.

## Solution

A Gymnasium-compatible RL environment simulating onboard constraints, with trained PPO agents that learn scheduling policies outperforming hand-crafted heuristics.

**Key differentiator**: Most satellite edge computing research addresses *where* to compute (offloading). We address *what* to compute given a fixed compute budget.

## Results

Benchmark across 4 scenarios (steady state, disaster surge, thermal throttling, communication blackout):

| Policy | Steady State Value | vs. Best Baseline |
|--------|-------------------|-------------------|
| **PPO (trained)** | ~35,000 | **+52%** |
| ValueDensity | 22,951 | — |
| RoundRobin | 20,928 | -8.8% |
| Priority | 17,190 | -25.1% |
| Random | 11,967 | -47.8% |
| FIFO | 6,507 | -71.6% |

PPO learns temporal patterns that myopic heuristics cannot exploit: prioritizing compression near ground contacts, batching high-value anomaly tasks, and conserving power for value bursts.

## Quickstart

```bash
# Setup
uv venv --python 3.12
source .venv/bin/activate
uv sync

# Run tests (93%+ coverage)
uv run pytest

# Run benchmark (quick mode: 5 seeds, 100 steps)
uv run scripts/run_phase4_benchmark.py --quick

# Train PPO agent
uv run scripts/train_phase2.py
```

## Project Structure

```
src/satellite_edge/
├── environment/           # Gymnasium RL environment
│   ├── satellite_env.py   # SatelliteEnv (14 or 22-dim obs, 5 actions)
│   └── tasks.py           # Task types, queue, value decay
├── orbital/               # Orbital mechanics
│   ├── orbit_propagator.py # Simplified + Skyfield SGP4
│   └── ground_station.py   # 6 real station coordinates
├── agents/                # RL agents (Phase 2)
│   ├── baselines.py       # 6 heuristic schedulers
│   ├── ppo_agent.py       # PPO wrapper (SB3)
│   └── evaluation.py      # Policy comparison
├── models/                # CV models (Phase 3)
│   ├── detector.py        # YOLOv8n (~80ms CPU)
│   ├── cloud_mask.py      # Lightweight U-Net (~55ms)
│   └── compressor.py      # Convolutional autoencoder (~10ms)
├── data/                  # Imagery pipeline
│   ├── tile_loader.py     # GeoTIFF loading
│   └── value_estimator.py # Cloud/entropy/edge metrics
└── evaluation/            # Benchmarking (Phase 4)
    ├── scenarios.py       # 4 benchmark scenarios
    ├── runner.py          # Multi-seed evaluation
    └── visualization.py   # Result plots
```

## RL Environment

**Observation** (14-dim base, 22-dim with image features):
- Compute available, buffer usage
- Queue depths and compute backlog per task type (4 types)
- Ground contact status, time to contact, contact duration
- Power available
- [Extended] Average cloud fraction and quality per task type

**Action** (Discrete, 5):
- 0-3: Prioritize task type (Detection, Anomaly, CloudMask, Compression)
- 4: Idle (save power)

**Reward**:
- +value × decay(age) when task completes
- +bonus during ground contact downlink
- -penalty for buffer overflow

## Benchmark Scenarios

| Scenario | Description | Stress Factor |
|----------|-------------|---------------|
| steady_state | Normal operations | 1.09x contention |
| event_surge | Disaster response | 2x arrival, 6x priority |
| degraded_compute | Thermal throttling | 50% capacity |
| long_blackout | Extended no-contact | Reduced windows |

## Hardware Basis

Simulation parameters grounded in real hardware:

- **Jiguang 1000** (space-grade): 32 TOPS
- **ESA PhiSat-2** Intel Myriad X: 4 TOPS, 2W
- **Target inference**: <100ms per 384×384 tile
- **Model budget**: ~40MB total (YOLOv8n + U-Net + autoencoder)

Orbital model uses ISS-like parameters: 400km altitude, 51.6° inclination, 90-minute period.

## Limitations

1. **Synthetic imagery** — Uses generated tiles rather than real Sentinel-2
2. **Simplified power model** — No battery state, constant solar input
3. **Single satellite** — No constellation coordination
4. **Pretrained models** — CV models not fine-tuned for satellite imagery

## Future Work

- Multi-satellite constellation with inter-satellite links
- Real imagery integration via SentinelHub/STAC
- Hierarchical RL for scene-level then tile-level scheduling
- Hardware-in-the-loop validation on edge accelerators

## References

1. [Priority-aware task offloading (MA2C)](https://link.springer.com/article/10.1007/s44443-025-00160-w)
2. [Multi-objective RL for satellite edge](https://www.nature.com/articles/s41598-025-10553-6)
3. [Orbit-aware task scheduling](https://link.springer.com/article/10.1007/s10586-025-05663-9)
4. [DRL task scheduling](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-023-00538-z)
5. [Computing over Space](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.06.005) — 3-4 order magnitude compute gap
6. [SatEdgeSim](https://ieeexplore.ieee.org/document/9139057) — Java-based simulation toolkit
7. [PPO](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
