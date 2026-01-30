# Related Work

Literature positioning and differentiation from existing research.

## Problem Space

Satellite edge computing sits at the intersection of:

1. **Edge computing**: Moving computation closer to data sources
2. **Satellite communications**: LEO constellations, inter-satellite links
3. **Reinforcement learning**: Sequential decision-making under uncertainty
4. **Computer vision**: On-sensor processing for Earth observation

The field is active, with most work focused on **task offloading** (deciding where to compute) rather than **onboard scheduling** (deciding what to compute given fixed resources).

## Our Differentiation

| Aspect | Existing Literature | This Work |
|--------|--------------------| -----------|
| **Decision** | Where to compute (ground vs satellite) | What to compute (given fixed onboard budget) |
| **Agent location** | Ground terminal or MEC server | Onboard satellite |
| **Resource** | Network bandwidth, offloading latency | Compute TOPS, buffer, power |
| **Workload** | Generic tasks | CV-specific (detection, anomaly, compression) |
| **Value model** | Static priorities | Time-decaying, content-aware |

Most papers model satellites as MEC (Mobile Edge Computing) servers receiving offloaded tasks from ground terminals. We model the satellite as the task generator, facing limited compute for its own EO workload.

## Key Papers

### Task Offloading

**1. Priority-aware task offloading (MA2C)**
- [Springer, 2025](https://link.springer.com/article/10.1007/s44443-025-00160-w)
- Multi-agent actor-critic for satellite-ground coordination
- Focus: Offloading decisions, not onboard scheduling
- Relevant: Priority modeling, multi-agent coordination patterns

**2. Multi-objective RL for satellite edge**
- [Nature Scientific Reports, 2025](https://www.nature.com/articles/s41598-025-10553-6)
- Balances latency, energy, and fairness
- Focus: Offloading with multiple objectives
- Relevant: Multi-objective reward design

**3. Orbit-aware task scheduling**
- [Springer Cluster Computing, 2025](https://link.springer.com/article/10.1007/s10586-025-05663-9)
- Incorporates orbital dynamics into scheduling
- Focus: Visibility windows in offloading decisions
- Relevant: Orbital mechanics integration

**4. DRL task scheduling minimizing processing time**
- [Journal of Cloud Computing, 2023](https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-023-00538-z)
- DQN for edge task scheduling
- Focus: Latency minimization
- Relevant: DRL formulation patterns

**5. MADDPG for LEO edge computing**
- [Computer Communications, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0140366424001828)
- Multi-agent DDPG for distributed satellite coordination
- Focus: Constellation-level offloading
- Relevant: Multi-satellite coordination (future work)

### Satellite-Specific

**6. Computing over Space**
- [Engineering, 2025](https://www.engineering.org.cn/engi/EN/10.1016/j.eng.2025.06.005)
- Survey of onboard computing capabilities
- Documents 3-4 order magnitude gap (space vs ground)
- Provides Jiguang 1000 specs (32 TOPS space-grade)
- **Key source for hardware grounding**

**7. Satellogic AI-first architecture**
- [Satellogic Blog, 2025](https://satellogic.com/2025/03/20/pushing-intelligence-to-the-edge-satellogics-vision-for-ai-powered-earth-observation/)
- Industry perspective on onboard AI
- Real deployment constraints and priorities

### RL Foundations

**8. Proximal Policy Optimization (PPO)**
- [arXiv, 2017](https://arxiv.org/abs/1707.06347)
- Schulman et al.
- Clipped surrogate objective for stable policy gradient
- Used for all RL experiments in this work

## Simulation Alternatives

### SatEdgeSim

- **Paper**: [IEEE, 2020](https://ieeexplore.ieee.org/document/9139057)
- **Code**: [GitHub](https://github.com/wjy491156866/SatEdgeSim)
- Java-based, extends PureEdgeSim
- Focus: Offloading simulation
- Why not used: Java integration overhead, offloading focus

### STK (Satellite Tool Kit)

- Industry standard for orbital mechanics
- Proprietary, expensive
- Overkill for RL research
- Why not used: Cost, complexity

### Skyfield

- **Docs**: [rhodesmill.org](https://rhodesmill.org/skyfield/)
- Python, open source
- Accurate SGP4 propagation
- **Used for orbital mechanics in this work**

### Gymnasium

- **Docs**: [Farama Foundation](https://gymnasium.farama.org/)
- Standard RL environment interface
- **Used as base for SatelliteEnv**

## Gap in Literature

Most satellite edge computing + RL papers share a common structure:

1. Ground terminal has task to compute
2. Decide: compute locally, offload to satellite, or offload to ground server
3. Optimize for latency, energy, or cost

This ignores a different problem: **satellites themselves generate massive data** (EO imagery) and must decide how to process it onboard before limited downlink windows.

Papers addressing this onboard scheduling problem are rare:
- Some industrial work (Satellogic, Planet) exists but isn't published
- Academic papers focus on offloading from ground
- No direct comparison of RL schedulers for onboard CV workloads

## Our Contribution

1. **Gymnasium environment** modeling onboard constraints (compute, buffer, power, ground contact)
2. **Content-aware task values** based on image metrics (cloud fraction, entropy, edges)
3. **PPO agent** outperforming heuristics by 52%
4. **Benchmark suite** across 4 operational scenarios
5. **Open-source** implementation for reproduction

## Future Directions

### Multi-Satellite Coordination

Extend to constellation scheduling:
- Inter-satellite links for task migration
- Distributed RL (MADDPG, MAPPO)
- Global optimization vs local greedy

### Hierarchical Scheduling

Two-level decision making:
- Scene-level: Which regions to image
- Tile-level: Which tiles to process (current work)

### Hardware-in-the-Loop

Validate on actual edge hardware:
- NVIDIA Jetson for development
- Intel Myriad X for space-representative
- Measure real inference times vs simulation

### Real Imagery Integration

Replace synthetic tiles:
- SentinelHub or STAC for Sentinel-2
- SpaceNet for labeled datasets
- Fine-tune CV models for satellite imagery

### Transfer Learning

Train on simulation, deploy on satellite:
- Domain randomization for robustness
- Sim-to-real gap analysis
- Online adaptation with limited onboard learning
