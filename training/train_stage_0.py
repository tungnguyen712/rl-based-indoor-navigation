"""
STAGE 0: Empty Room (Sanity Check - Can Agent Even Navigate at All?)
Goal: Move from spawn to goal in empty rectangular room
Map: 5x5 to 6x5, no obstacles except boundary walls
Target: ≥90% success rate (if this fails, robot/reward is broken)
Training: 1M timesteps (should be trivial)
"""
import os
import sys
import json
import gc
import torch as th
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from envs.indoor_maze_env import IndoorMazeEnv

# ==================== STAGE 0 HYPERPARAMETERS ====================
NUM_ENVS = 6  # Reduced from 16 - fixes CPU thrashing
POLICY = "MlpPolicy"
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256  # increased for stable gradients
N_EPOCHS = 4  # reduced to prevent overfitting
GAMMA = 0.99  # back to standard
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.05  # Moderate exploration - with heading reward, agent needs less random exploration
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Balanced network - not too large to prevent overfitting
POLICY_KWARGS = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Reduced from 256 for stability
    activation_fn=nn.ReLU
)

# Stage 0 specific
TOTAL_TIMESTEPS = 2_000_000  # 2M should be enough with proper episode length
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 30  # increased for more reliable evaluation
CHECKPOINT_FREQ = 250_000
USE_LR_SCHEDULE = True  # Enable learning rate decay for late-stage stability

def load_mazes(folder_path):
    maze_layouts = []
    for f in os.listdir(folder_path):
        if f.endswith(".json"):
            path = os.path.join(folder_path, f)
            with open(path, "r") as fp:
                maze_layouts.append(json.load(fp))
    return maze_layouts

def make_env(maze_layouts, render_mode="direct", **env_kwargs):
    def _init():
        env = IndoorMazeEnv(maze_layouts=maze_layouts, render_mode=render_mode, **env_kwargs)
        env = Monitor(env)
        return env
    return _init

class GarbageCollectionCallback(BaseCallback):
    def __init__(self, gc_freq=10000):
        super().__init__()
        self.gc_freq = gc_freq
    
    def _on_step(self):
        if self.n_calls % self.gc_freq == 0:
            gc.collect()
        return True

def main():
    print("\n" + "="*70)
    print("🟢 STAGE 0: EMPTY ROOM - SANITY CHECK")
    print("="*70)
    print("Goal: Navigate to goal in empty rectangular room (NO obstacles)")
    print("Target: ≥90% success rate (this should be TRIVIAL)")
    print("If this fails, the robot/reward system is fundamentally broken")
    print("="*70 + "\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_maze_dir = os.path.join(project_root, "assets", "train", "stage_0_empty")
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_0_empty")
    models_dir = os.path.join(project_root, "models", "stage_0")
    logs_dir = os.path.join(project_root, "logs", "stage_0")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    train_mazes = load_mazes(train_maze_dir)
    eval_mazes = load_mazes(eval_maze_dir)
    
    print(f"Loaded {len(train_mazes)} training mazes (empty rooms)")
    print(f"Loaded {len(eval_mazes)} evaluation mazes (empty rooms)\n")

    device = "cpu"

    train_env = SubprocVecEnv([
        make_env(train_mazes, render_mode="direct", terminate_on_collision=False)
        for _ in range(NUM_ENVS)
    ])
    eval_env = DummyVecEnv([
        make_env(eval_mazes, render_mode="direct", terminate_on_collision=True)
    ])

    print("Starting fresh training for Stage 0 with differential drive...")
    
    # Optional: Learning rate schedule for late-stage stability
    def lr_schedule(progress_remaining):
        """Decay LR from 3e-4 to 1e-5 over training"""
        return LEARNING_RATE * progress_remaining
    
    model = PPO(
        POLICY, train_env,
        policy_kwargs=POLICY_KWARGS,  # Use larger network
        learning_rate=lr_schedule if USE_LR_SCHEDULE else LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENTROPY_COEF,
        vf_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=logs_dir,
        device=device
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=EVAL_FREQ // NUM_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // NUM_ENVS,
        save_path=models_dir,
        name_prefix="stage_0_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    gc_callback = GarbageCollectionCallback(gc_freq=10000)
    
    print("Starting Stage 0 training...")
    print(f"Monitor: tensorboard --logdir {logs_dir}\n")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback, gc_callback],
        progress_bar=True
    )
    
    final_model_path = os.path.join(models_dir, "stage_0_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("✅ STAGE 0 COMPLETE!")
    print("="*70)
    print(f"Best model: {os.path.join(models_dir, 'best_model.zip')}")
    print(f"Final model: {final_model_path}")
    print("\n📊 Check TensorBoard:")
    print("   - If success rate ≥90%: System works, proceed to Stage A")
    print("   - If success rate <50%: Robot/reward is fundamentally broken")
    print("="*70 + "\n")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
