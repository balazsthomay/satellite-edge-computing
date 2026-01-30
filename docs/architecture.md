# System Architecture

Technical design of the satellite edge compute scheduler.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SATELLITE (LEO ~400km)                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         ONBOARD COMPUTE (32 TOPS)                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │  DETECTION  │  │   ANOMALY   │  │ CLOUD MASK  │  │ COMPRESSION │   │  │
│  │  │   6 TOPS    │  │   3 TOPS    │  │  1.5 TOPS   │  │   4 TOPS    │   │  │
│  │  │  Value: 10  │  │  Value: 15  │  │  Value: 3   │  │  Value: 5   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      ▲                                      │
│                                      │ RL Agent selects                     │
│                                      │ which to prioritize                  │
│  ┌───────────────────────────────────┴───────────────────────────────────┐  │
│  │                           TASK QUEUE (max 50)                         │  │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐                  │  │
│  │  │ D  │ A  │ D  │ C  │ CM │ D  │ A  │ C  │ D  │ ...│  Tasks arrive    │  │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘  via Poisson     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         BUFFER (256 GB)                               │  │
│  │  ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (35% full)       │  │
│  │  Stores task inputs + outputs until downlink                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      │ Downlink when visible                │
│                                      ▼                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ ~15% of orbit
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GROUND STATIONS                                   │
│   Svalbard    Fairbanks    Wallops    Hartebeesthoek    Alice Springs       │
│   McMurdo                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## RL Environment Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SatelliteEnv                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OBSERVATION (14-dim base, normalized [0,1])                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ [0]  compute_available    How much compute left this step           │    │
│  │ [1]  buffer_usage         Fraction of 256GB used                    │    │
│  │ [2]  queue_depth_DET      Detection tasks waiting                   │    │
│  │ [3]  queue_depth_ANO      Anomaly tasks waiting                     │    │
│  │ [4]  queue_depth_CM       Cloud mask tasks waiting                  │    │
│  │ [5]  queue_depth_COMP     Compression tasks waiting                 │    │
│  │ [6]  queue_compute_DET    TOPS needed for detection                 │    │
│  │ [7]  queue_compute_ANO    TOPS needed for anomaly                   │    │
│  │ [8]  queue_compute_CM     TOPS needed for cloud mask                │    │
│  │ [9]  queue_compute_COMP   TOPS needed for compression               │    │
│  │ [10] ground_contact       1 if visible to station, 0 otherwise      │    │
│  │ [11] time_to_contact      Normalized time until next pass           │    │
│  │ [12] contact_duration     Remaining contact time (if in contact)    │    │
│  │ [13] power_available      Fraction of 300W remaining                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  EXTENDED OBSERVATION (22-dim, when use_image_features=True)                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ [14:18] avg_cloud[4]      Mean cloud fraction per task type         │    │
│  │ [18:22] avg_quality[4]    Mean image quality multiplier per type    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ACTION (Discrete, 5 choices)                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 0 = Prioritize DETECTION     (ships, vehicles, aircraft)            │    │
│  │ 1 = Prioritize ANOMALY       (fires, floods, changes)               │    │
│  │ 2 = Prioritize CLOUD_MASK    (preprocessing for others)             │    │
│  │ 3 = Prioritize COMPRESSION   (prepare for downlink)                 │    │
│  │ 4 = IDLE                     (save power for later)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  REWARD                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ + task_value × decay(age)    When task completes                    │    │
│  │ + 0.1 × data_downlinked      During ground contact                  │    │
│  │ - 2.0 × buffer_overflow      If buffer exceeds capacity             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Task System

Four task types model different onboard CV workloads:

| Task Type | Compute (TOPS) | Memory (GB) | Base Value | Deadline Sensitivity |
|-----------|---------------|-------------|------------|---------------------|
| DETECTION | 6.0 | 2.0 | 10 | 0.8 |
| ANOMALY | 3.0 | 1.5 | 15 | 0.9 (most urgent) |
| CLOUD_MASK | 1.5 | 0.5 | 3 | 0.3 (least urgent) |
| COMPRESSION | 4.0 | 1.0 | 5 | 0.5 |

### Value Decay

Task value decays exponentially with age, modeling time-sensitivity of information:

```
    Value
      │
  15 ─┤ ─ ─ ─ ─╮ ANOMALY (high urgency, fast decay)
      │         ╲
  10 ─┤ ─ ─ ─ ─ ─╲─ ─ ─╮ DETECTION
      │           ╲     ╲
   5 ─┤ ─ ─ ─ ─ ─ ─╲─ ─ ─╲─ ─ ─ ─ ─ ─ ─ ─╮ COMPRESSION
      │             ╲     ╲               ╲
   3 ─┤ ─ ─ ─ ─ ─ ─ ─╲─ ─ ─╲─ ─ ─ ─ ─ ─ ─ ─╲─ ─ CLOUD_MASK (low urgency)
      │               ╲     ╲               ╲
   0 ─┼───────────────┴─────┴───────────────┴───────────►
      0              10     20             40          Age (steps)

decay = exp(-deadline_sensitivity × age / 10)
```

This creates the core scheduling challenge: high-value tasks (ANOMALY) lose value fastest, but lower-value tasks (CLOUD_MASK) may be worth processing if ANOMALY tasks have already aged.

### Image-Aware Tasks (Phase 3)

`ImageTask` extends `Task` with content metrics:

```python
@dataclass
class ImageTask(Task):
    tile_path: str | None = None
    cloud_fraction: float = 0.0      # [0,1] from brightness threshold
    entropy_score: float = 0.5       # [0,1] local Shannon entropy
    edge_density: float = 0.5        # [0,1] Sobel magnitude

    @property
    def image_quality_multiplier(self) -> float:
        # Task-specific value modulation
        match self.task_type:
            case TaskType.DETECTION:
                return (1 - cloud_fraction) * (0.3*entropy + 0.7*edges)
            case TaskType.ANOMALY:
                return (1 - cloud_fraction) * (0.5*entropy + 0.5*edges)
            case TaskType.CLOUD_MASK:
                return 0.2 + 0.8*cloud_fraction  # More clouds = more useful
            case TaskType.COMPRESSION:
                return 0.3 + 0.7*entropy  # High entropy = better compression
```

## Orbital Model

Two implementations for different fidelity needs:

### Simplified Model (default)

Sinusoidal approximation for fast training:

```
    Orbital Period: 5400s (90 min)          Contact: ~15% of orbit

                          ╭─────────────╮
                       ╱                 ╲
                     ╱    ┌─────────┐      ╲
                    │     │ CONTACT │       │
                    │     │ WINDOW  │       │
                   │      │ (10min) │        │
                   │      └─────────┘        │
                   │                         │
    ◄──────────────┼─────────────────────────┼──────────────►
    Orbit start    │                         │           Orbit end
                   │      ┌─────────┐        │
                   │      │ CONTACT │        │
                    │     │ WINDOW  │       │
                    │     │ (10min) │       │
                     ╲    └─────────┘      ╱
                       ╲                 ╱
                          ╰─────────────╯
```

### Full Skyfield Model

SGP4 propagation with real TLE:
- Inclination: 51.6° (ISS-like)
- Altitude: ~400km
- Real ground station coordinates
- Computes actual elevation angles

## CV Pipeline (Phase 3)

```
┌────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  GeoTIFF Tile  │────▶│  ValueEstimator  │────▶│    ImageTask       │
│  (384×384×4)   │     │                  │     │                    │
│                │     │  • cloud_frac    │     │  • tile_path       │
│  RGB + NIR     │     │  • entropy       │     │  • cloud_fraction  │
│                │     │  • edge_density  │     │  • entropy_score   │
└────────────────┘     └──────────────────┘     │  • edge_density    │
                                               │  • quality_mult    │
                                               └─────────┬──────────┘
                                                         │
                                                         ▼
                        ┌────────────────────────────────────────────┐
                        │              CV Models (Edge-Ready)        │
                        │                                            │
                        │  SatelliteDetector  │ CloudMaskModel │ ImageCompressor
                        │  YOLOv8n            │ U-Net          │ Autoencoder
                        │  3.2M params        │ ~1M params     │ ~0.5M params
                        │  ~80ms CPU          │ ~55ms CPU      │ ~10ms CPU
                        └────────────────────────────────────────────┘
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CONTENTION ENVIRONMENT                           │    │
│  │         compute_capacity=4 TOPS  |  task_arrival_rate=12.0/step     │    │
│  │         buffer=256 GB  |  power=300W  |  episode=500 steps          │    │
│  │                                                                     │    │
│  │         Contention ratio: 1.09 (demand > capacity → backlog)        │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    VecNormalize + DummyVecEnv (×8)                  │     │
│  │                                                                    │     │
│  │  norm_obs=True  |  norm_reward=True  |  γ=0.995  |  clip=10.0     │     │
│  └────────────────────────────────┬───────────────────────────────────┘     │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                          PPO (MlpPolicy)                           │     │
│  │                                                                    │     │
│  │  Actor [64,64] Tanh + ortho  │  Critic [64,64] Tanh + ortho       │     │
│  │  14-dim obs ─► 5 actions     │  14-dim obs ─► V(s)                │     │
│  │                                                                    │     │
│  │  lr=3e-4  |  n_steps=1024  |  batch=64  |  epochs=10              │     │
│  │  clip=0.2  |  ent_coef=0.01  |  target_kl=0.03                    │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Baseline Schedulers

All baselines implement a `select_action(obs) → int` interface:

| Scheduler | Strategy | Performance |
|-----------|----------|-------------|
| **ValueDensity** | Maximize value/TOPS ratio | Best heuristic |
| RoundRobin | Cycle through types equally | -8.8% |
| Priority | depth × value_rate weighting | -25.1% |
| GreedyCompute | Highest compute backlog first | -55.8% |
| Random | Uniform random (includes IDLE) | -47.8% |
| FIFO | Deepest queue first | -71.6% (worst) |

## Why PPO Beats ValueDensity by 52%

ValueDensity is a *myopic* heuristic—it greedily picks the highest value/compute ratio each step without considering:

1. **Temporal patterns**: PPO learns to prioritize COMPRESSION near ground contacts for downlink
2. **Queue composition**: PPO batches high-value ANOMALY tasks optimally
3. **Power state**: PPO observes `power_available` and throttles during low-power periods
4. **Non-myopic planning**: PPO's value function V(s) estimates future returns, enabling short-term sacrifices for long-term gains

## Code Structure

```
src/satellite_edge/
│
├── __init__.py
│
├── environment/
│   ├── satellite_env.py        # SatelliteEnv (Gymnasium)
│   │   ├── SatelliteConfig     #   Resource limits, image features flag
│   │   ├── EpisodeConfig       #   Episode structure
│   │   └── SatelliteEnv        #   Main environment class
│   │
│   └── tasks.py                # Task system
│       ├── TaskType            #   Enum (DET, ANO, CM, COMP)
│       ├── TaskSpec            #   compute_cost, memory, value, decay
│       ├── Task                #   Individual task instance
│       ├── ImageTask           #   Task with image metrics
│       └── TaskQueue           #   Priority queue + aggregation
│
├── orbital/
│   ├── ground_station.py       # Ground stations
│   │   ├── GroundStation       #   Name, lat, lon, elevation, rate
│   │   └── GroundStationNetwork#   6 real stations
│   │
│   └── orbit_propagator.py     # Orbital mechanics
│       ├── SimplifiedOrbitModel#   Fast sinusoidal approximation
│       └── OrbitPropagator     #   Full SGP4 via Skyfield
│
├── agents/
│   ├── baselines.py            # 6 heuristic schedulers
│   ├── ppo_agent.py            # PPO training wrapper
│   ├── evaluation.py           # Policy comparison
│   └── training.py             # Training orchestration
│
├── models/
│   ├── base.py                 # BaseCVModel ABC
│   ├── detector.py             # YOLOv8n wrapper
│   ├── cloud_mask.py           # Lightweight U-Net
│   └── compressor.py           # Convolutional autoencoder
│
├── data/
│   ├── tile_loader.py          # GeoTIFF loading
│   ├── value_estimator.py      # Cloud/entropy/edge metrics
│   └── datasets.py             # Synthetic tile generation
│
└── evaluation/
    ├── scenarios.py            # 4 benchmark scenarios
    ├── metrics.py              # IQM, confidence intervals
    ├── runner.py               # Multi-seed evaluation
    └── visualization.py        # Result plots
```
