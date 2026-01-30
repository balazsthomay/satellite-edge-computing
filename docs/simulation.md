# Simulation Grounding

How the simulation maps to real satellite hardware, orbital mechanics, and operational constraints.

## Hardware Basis

### Space-Grade Compute

The simulation's compute constraints are grounded in real hardware:

| Hardware | TOPS | Power | Source |
|----------|------|-------|--------|
| **Jiguang 1000** (space-grade) | 32 | N/A | [Computing over Space](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.06.005) |
| **Intel Myriad X** (ESA PhiSat-2) | 4 | 2W | ESA mission specs |
| **NVIDIA Jetson** (COTS, not rad-hard) | 32 | 15W | Consumer reference |

The 3-4 order magnitude gap between space-grade and ground hardware comes from radiation hardening requirements. Single-event upsets (SEUs) from cosmic rays require either:
- Radiation-hardened (rad-hard) components with older process nodes
- Triple modular redundancy (TMR) with voting logic
- Error-correcting memory and watchdog timers

### Default Configuration

```python
SatelliteConfig(
    compute_capacity=4.0,      # TOPS (PhiSat-2 class)
    buffer_capacity=256.0,     # GB onboard storage
    power_capacity=300.0,      # Watts available
    max_queue_size=50,         # Task queue depth
)
```

The contention environment uses `compute_capacity=4` with `task_arrival_rate=12` to create ~1.09x oversubscription, forcing scheduling decisions.

## Orbital Parameters

### ISS-like Orbit

```
Parameter          Value           Rationale
─────────────────────────────────────────────────────────
Altitude           ~400 km         Typical LEO for EO satellites
Inclination        51.6°           ISS orbit, good coverage
Period             ~90 min         Standard LEO
Eccentricity       ~0              Circular orbit assumed
```

### Ground Contact Model

Real satellites have ~15% ground station visibility per orbit. The simulation models this with:

**Simplified Model** (default):
```python
# Contact windows at fixed orbit phases
contact_phases = [0.15, 0.65]  # Two passes per orbit
contact_duration = 600         # 10 minutes each
```

**Full Skyfield Model**:
- SGP4 propagation from real TLE
- Elevation angle computation to each ground station
- Contact when elevation > minimum (typically 5-10°)

## Ground Station Network

Six real ground stations with actual coordinates:

| Station | Location | Latitude | Longitude | Min Elevation |
|---------|----------|----------|-----------|---------------|
| Svalbard | Norway | 78.23° | 15.39° | 5° |
| Fairbanks | Alaska | 64.86° | -147.85° | 5° |
| Wallops | Virginia | 37.94° | -75.47° | 10° |
| Hartebeesthoek | South Africa | -25.89° | 27.69° | 10° |
| Alice Springs | Australia | -23.70° | 133.87° | 10° |
| McMurdo | Antarctica | -77.85° | 166.67° | 5° |

This network provides global coverage with polar stations (Svalbard, McMurdo) capturing high-inclination passes.

## Task Type Specifications

### Compute Costs

Based on typical edge inference workloads:

| Task | Compute (TOPS) | Memory (GB) | Model Basis |
|------|---------------|-------------|-------------|
| DETECTION | 6.0 | 2.0 | YOLOv8n (~8.7 GFLOPs) |
| ANOMALY | 3.0 | 1.5 | Lightweight change detection |
| CLOUD_MASK | 1.5 | 0.5 | Small U-Net (~1 GFLOPs) |
| COMPRESSION | 4.0 | 1.0 | Autoencoder (~0.5 GFLOPs) |

The TOPS values are intentionally higher than raw FLOP counts to account for:
- Memory bandwidth limitations
- I/O overhead (reading tiles from storage)
- Thermal constraints reducing sustained throughput

### Value and Urgency

| Task | Base Value | Deadline Sensitivity | Rationale |
|------|------------|---------------------|-----------|
| ANOMALY | 15 | 0.9 | Fires, floods need immediate response |
| DETECTION | 10 | 0.8 | Ship/vehicle tracking is time-sensitive |
| COMPRESSION | 5 | 0.5 | Prepare for downlink, moderate urgency |
| CLOUD_MASK | 3 | 0.3 | Preprocessing, can wait |

### Task Arrival Model

Tasks arrive via Poisson process with weighted type distribution:

```python
type_weights = {
    DETECTION: 0.35,    # Most common
    ANOMALY: 0.10,      # Rare but high-value
    CLOUD_MASK: 0.25,   # Frequent preprocessing
    COMPRESSION: 0.30,  # Regular downlink prep
}
```

5% of tasks receive a priority multiplier (2-5×), simulating urgent requests from ground operators.

## CV Model Specifications

### Target Hardware

ESA PhiSat-2 mission parameters:
- Intel Myriad X VPU: 4 TOPS, 2W
- Target inference: <100ms per 384×384 tile
- Total model budget: ~50MB

### Implemented Models

| Model | Parameters | Size | CPU Time | Output |
|-------|------------|------|----------|--------|
| SatelliteDetector (YOLOv8n) | 3.2M | ~6MB | ~80ms | Bounding boxes |
| CloudMaskModel (U-Net) | ~1M | ~5MB | ~55ms | Binary mask |
| ImageCompressor (CAE) | ~0.5M | ~2MB | ~10ms | Latent + reconstruction |

**Total: ~40MB, ~145ms sequential** — within PhiSat-2 constraints.

### Image Value Estimation

Content-aware task values based on image metrics:

```python
# Cloud detection: brightness threshold
cloud_fraction = (pixels > 200).mean()

# Scene complexity: local Shannon entropy
entropy_score = skimage.filters.rank.entropy(gray, disk(5)).mean() / 8

# Feature richness: Sobel edge detection
edge_density = np.percentile(sobel_magnitude, 95)
```

Different task types weight these differently:
- DETECTION: Penalized by clouds, boosted by edges
- ANOMALY: Penalized by clouds, boosted by entropy + edges
- CLOUD_MASK: Boosted by clouds (more useful when cloudy)
- COMPRESSION: Boosted by entropy (high entropy = good candidate)

## Validation Approach

### What's Validated

1. **Orbital mechanics**: Skyfield propagation matches STK within meters
2. **Task queue dynamics**: Proper Poisson arrival, value decay
3. **Constraint satisfaction**: Compute/power/buffer limits enforced
4. **Gymnasium compliance**: Passes `gymnasium.utils.env_checker`

### What's Approximated

1. **Power model**: Simplified constant solar input, no battery state
2. **Thermal model**: No throttling based on actual temperature
3. **Communication**: Fixed downlink rate, no packet loss or weather

### What's Synthetic

1. **Imagery**: Generated tiles, not real Sentinel-2
2. **Task content**: Random image metrics, not from actual scenes
3. **Ground truth**: No labeled objects for detection evaluation

## Known Limitations

### Single Satellite

No inter-satellite links or constellation coordination. Real LEO constellations (Starlink, Planet) coordinate across satellites.

### Simplified Power

No battery state-of-charge, eclipse periods, or solar panel pointing. Real satellites have complex power management with eclipse survival modes.

### No Attitude Control

Assumes nadir-pointing. Real satellites have slew constraints and may need to point away from target for thermal management.

### No Data Prioritization

All completed tasks assumed equal for downlink. Real operations prioritize high-value data for limited bandwidth.

## Comparison to SatEdgeSim

[SatEdgeSim](https://github.com/wjy491156866/SatEdgeSim) is a Java-based simulator extending PureEdgeSim. Key differences:

| Aspect | SatEdgeSim | This Simulation |
|--------|------------|-----------------|
| Focus | Task offloading (where) | Onboard scheduling (what) |
| Language | Java | Python |
| RL integration | External | Native Gymnasium |
| Orbital fidelity | Basic | Skyfield SGP4 |
| Imagery | None | Synthetic + pipeline |

We chose to build a lightweight Python environment for tighter RL integration rather than wrapping SatEdgeSim.

## Future Fidelity Improvements

1. **Real imagery**: SentinelHub/STAC integration for actual Sentinel-2 tiles
2. **Battery model**: Eclipse periods, charge cycles, power budgeting
3. **Thermal model**: CPU throttling based on thermal state
4. **Constellation**: Multi-satellite coordination via shared queue
5. **Communication model**: Variable link quality, weather effects
