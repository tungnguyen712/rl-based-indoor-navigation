# RL-based Indoor Navigation Robot Simulation

### Demo Videos

**Demo 1**

<video src="videos/9x9_hard_mazes/episode_1.mp4" controls></video>

**Demo 2**

<video src="videos/9x9_hard_mazes/episode_8.mp4" controls></video>

### Overview
This project implements a learning-based indoor navigation system using PyBullet and PPO. A mobile robot must navigate from a random start to a random goal inside an indoor maze-like environment, using only local lidar observations.

### Sensors
1. 2d Lidar (ray casting)
2. Contact sensors (collision detection for penalty)

### Observation space (71 dimensions)
[64 lidar distances (normalized), goal_distance (normalized), goal_angle (normalized), velocity (3), prev_action (2)]

### Task
At each episode, the robot will spawn at a random valid location. The goal position is randomly sampled such that there is a valid path between the starting and goal point. The starting point and goal point are guaranteed to not spawn too close together. The robot will attempt to navigate toward the goal while avoiding walls.

### Reward
**Terminal:**
- +10.0 for reaching goal (within 0.5m radius)
- -5.0 for wall collision

**Step-based:**
- Distance shaping: +1.0 × (distance improvement toward goal)
- Time penalty: -0.01 per step
- Backward penalty: -0.5 when driving backward

### Setup and Installation
1. **Clone the repository:**
```bash
git clone <repository-url>
cd rl-based-indoor-navigation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training (Staged Curriculum)

**Stage 0 (7×7 no walls):**
```bash
python training/train_stage_0.py
```

**Stage A (7×7 foundation):**
```bash
python training/train_stage_a.py
```

**Stage B (9×9 medium):**
```bash
python training/train_stage_b.py
```

**Stage C (9×9 hard):**
```bash
python training/train_stage_c.py
```

**Monitor training:**
```bash
tensorboard --logdir=logs
```

### Evaluation

**Test on Stage C (final target):**
```bash
python eval/diagnose_stage_c.py
```
Runs 10 episodes with GUI. Success videos saved to `videos/9x9_hard_mazes/`.
