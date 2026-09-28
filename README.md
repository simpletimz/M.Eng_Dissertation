## M.Eng_Dissertation
## The public repo for my Master dissertation only readme is available until Paper is published.

This repository implements, benchmarks, and evaluates Deep Reinforcement Learning (DRL) algorithms for optimizing Random Early Detection (RED) queue weights ($w_q$) in network bottlenecks using the NS-3 discrete-event network simulator.

---

## Algorithms Implemented

1. **DQN (Deep Q-Network):** Standard baseline utilizing experience replay and target network decoupling.
2. **DDQN (Double Deep Q-Network):** Decouples action selection from action evaluation to eliminate overestimation bias under noisy network jitter.
3. **DDQN-PER (Double DQN with Prioritized Experience Replay):** Focuses learning on critical congestion transitions via SumTree TD-error prioritization and Importance Sampling correction.

---

## Project Directory Structure

```text
project_root/
├── requirements.txt             # Primary DRL & simulation dependencies (PyTorch 2.6.0, Gym, PyZMQ)
├── requirements_tensorboard.txt # Dedicated lightweight TensorBoard environment (Python 3.12.9)
├── README.md                    # System setup, build guide, and execution instructions
│
├── env_gym/                     # NS-3 Gymnasium Environment Interface
│   ├── env.py                   # Top-level Gym environment wrapper (NetworkCongestionEnv)
│   ├── env_state.py             # Telemetry dataclass (SimulationState) & JSON parser
│   ├── env_features.py          # Feature normalization & multi-objective reward calculation
│   ├── zmq_bridge.py            # IPC socket lifecycle, handshake protocol, & message transport
│   └── ns3_launcher.py          # NS-3 subprocess execution, CLI injection, & log redirection
│
├── training_code/               # Reinforcement Learning Training Code
│   ├── base_agent.py            # Base agent scaffolding (loops, logging, validation, checkpoints)
│   ├── network.py               # 128-128 MLP Q-Network architecture
│   ├── replay_buffers.py        # Uniform deque buffer & SumTree prioritized buffer
│   ├── metadata.py              # System hardware specs & duration tracking
│   ├── dqn_agent.py             # Standard DQN Bellman optimization
│   ├── ddqn_agent.py            # Decoupled Double-DQN Bellman optimization
│   ├── ddqnper_agent.py         # Double-DQN + PER priority & IS weighting
│   ├── train_dqn.py             # Sequential runner for DQN baseline (seeds 100-500)
│   ├── train_ddqn.py            # Parallel Pool runner for DDQN (5 concurrent seeds)
│   └── train_ddqnper.py         # Parallel Pool runner for DDQN-PER (5 concurrent seeds)
│
├── plots/                       # Top-Level Visualization Directory
│   ├── plot_dqn/
│   │   └── utils.py             # DQN convergence trends, boxplots, and heatmaps
│   ├── plot_ddqn/
│   │   └── utils.py             # DDQN convergence trends, boxplots, and heatmaps
│   └── plot_ddqnper/
│       └── utils.py             # DDQN-PER convergence trends, boxplots, and heatmaps
│
└── results/                     # AUTOMATICALLY GENERATED OUTPUTS
    ├── dqn/
    │   ├── csv/                 # train_stats_S*.csv, test_stats_S*.csv, action_history_S*.csv, system_metadata.csv
    │   ├── checkpoints/         # best_model_S*.pth
    │   ├── tensorboard/         # Event files for real-time tracking
    │   ├── logs/                # ns3_gym_output.log
    │   └── plots/               # Publication-ready figures
    ├── ddqn/
    │   ├── csv/
    │   ├── checkpoints/
    │   ├── tensorboard/
    │   ├── logs/                # ns3_log_Seed*.log (isolated parallel logs)
    │   └── plots/
    └── ddqnper/
        ├── csv/
        ├── checkpoints/
        ├── tensorboard/
        ├── logs/                # ns3_log_Seed*.log
        └── plots/
```

---

## 1. System Requirements & Dependencies

* **Operating System:** Linux (Ubuntu 22.04 LTS / Debian / Kali Linux recommended)
* **Simulator:** Network Simulator 3 (NS-3.43)
* **Python Runtime:** Python 3.10 – 3.12
* **Hardware:** Multi-core x86_64 CPU (minimum 6 cores recommended for parallel trials)

---

## 2. NS-3 Simulator Installation (Step-by-Step)

### Step 2.1: Install Linux Build Prerequisites
Open a terminal and install all required compilers, build utilities, and ZeroMQ development headers:
```bash
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    g++ \
    gcc \
    python3-dev \
    libzmq3-dev \
    pkg-config \
    sqlite3 \
    libsqlite3-dev \
    libxml2 \
    libxml2-dev
```

### Step 2.2: Download and Unpack NS-3.43
```bash
cd ~
wget https://www.nsnam.org/release/ns-allinone-3.43.tar.bz2
tar -xjf ns-allinone-3.43.tar.bz2
cd ns-allinone-3.43/ns-3.43/
```

### Step 2.3: Configure and Build NS-3 in Optimized Mode
Configure NS-3 for maximum execution throughput:
```bash
./ns3 configure -d optimized --enable-examples --enable-tests
./ns3 build
```

### Step 2.4: Deploy and Build All 7 Simulation Scripts in NS-3 Scratch
Copy all simulation source files into the NS-3 `scratch/` directory:

1. **DRL Training & Testing:**
   * `baselinered.cc`
2. **In-Distribution Baselines:**
   * `red.cc`
   * `droptail.cc`
   * `codel.cc`
3. **Out-of-Distribution (OOD) Static Baselines:**
   * `staticred.cc`
   * `staticdroptail.cc`
   * `staticcodel.cc`

```bash
cd /home/kali/ns-allinone-3.43/ns-3.43/

# Copy all .cc files from your workspace into NS-3 scratch
cp /path/to/your/project/ns3_scratch/*.cc scratch/

# Build all binaries in optimized mode
./ns3 build
```

Verify that all 7 standalone binaries exist in `build/scratch/`:
```bash
ls -l build/scratch/ns3.43-*-optimized
```

You should see:
* `ns3.43-baselinered-optimized`
* `ns3.43-red-optimized`
* `ns3.43-droptail-optimized`
* `ns3.43-codel-optimized`
* `ns3.43-staticred-optimized`
* `ns3.43-staticdroptail-optimized`
* `ns3.43-staticcodel-optimized`

---

## 3. Python Environment Setup

### Option A: Main Training Environment (PyTorch 2.6.0, Gym, PyZMQ)
Use this environment to run training, validation, zero-shot testing, and plotting:
```bash
cd project_root
python3 -m venv venv_drl
source venv_drl/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Dedicated TensorBoard Visualization Environment
If running a separate monitoring shell on Python 3.12.9:
```bash
python3 -m venv venv_tb
source venv_tb/bin/activate
pip install -r requirements_tensorboard.txt
```

---

## 4. How to Run Experiments

Always execute training runs from inside the `training_code/` folder:
```bash
cd training_code
```

### 1. Run DQN Baseline (Strictly Sequential Execution)
Trains across 5 independent seeds (100, 200, 300, 400, 500) sequentially using a single environment instance on port `5555`:
```bash
python train_dqn.py
```

### 2. Run DDQN (Parallel Multiprocessing Execution)
Launches 5 concurrent trials using `multiprocessing.Pool` across CPU cores on isolated ports (`5556`–`5560`) with dedicated seed prefixes:
```bash
python train_ddqn.py
```

### 3. Run DDQN-PER (Parallel Prioritized Experience Replay)
Executes Double-DQN with SumTree prioritized experience replay and Importance Sampling across 5 parallel workers:
```bash
python train_ddqnper.py
```

---

## 5. Automated Data Routing & Results Structure

All metrics, model checkpoints, simulation logs, and generated figures are automatically routed to `results/<algorithm>/` to prevent overwriting:

* `results/<algo>/csv/`: Contains `train_stats_S*.csv`, `test_stats_S*.csv`, `action_history_S*.csv`, and `system_metadata.csv`.
* `results/<algo>/checkpoints/`: Contains `best_model_S*.pth` saved during evaluation phases.
* `results/<algo>/tensorboard/`: Contains event logs tracking rewards, Huber losses, TD-errors, and action selections.
* `results/<algo>/logs/`: Contains simulator standard output logs (`ns3_log_Seed*.log` / `ns3_gym_output.log`).
* `results/<algo>/plots/`: Contains generated publication figures:
  * `Q1_Trend_*.png`: Shaded line plots showing mean performance $\pm 1$ standard deviation.
  * `Q1_Boxplot_Generalization.png`: Zero-shot throughput distributions across unseen network topologies.
  * `Q1_Heatmap_Policy_Evolution_S*.png`: Action probability transitions across training episodes.
  * `Q1_Action_Global_Preference.png`: Aggregate decision frequency distribution.

---

## 6. Real-Time Monitoring & Manual Plotting

### Live TensorBoard Dashboard
To view real-time learning metrics across all algorithms:
```bash
tensorboard --logdir=results/
```
Open `http://localhost:6006` in your web browser.

### Manually Re-generating Plots
If you modify plot parameters or wish to regenerate figures without retraining, run the plotting script from inside its specific folder:
`project_root`:
```bash
# For DQN figures
python plots/plot_dqn/utils.py

# For DDQN figures
python plots/plot_ddqn/utils.py

# For DDQN-PER figures
python plots/plot_ddqnper/utils.py
```
