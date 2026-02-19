"""
STAGE A: Foundation (Learn Not to Crash + Turn)
Goal: Basic steering, obstacle avoidance, turning in corridors
Map: 7×7, wide 2-cell corridors, no choke points
Target: ≥60% success rate
Training: 3M timesteps
"""
import os
import sys
import json
import gc
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from envs.indoor_maze_env import IndoorMazeEnv

# ==================== STAGE A HYPERPARAMETERS ====================
NUM_ENVS = 4  # Reduced for stability
POLICY = "MlpPolicy"
LEARNING_RATE = 3e-4  # Standard
N_STEPS = 2048
BATCH_SIZE = 256  # Match Stage 0
N_EPOCHS = 4      # Reduced from 10 to 4 (Prevent overfitting to "stuck" batches)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01  # Attempt 4 setting (higher exploration than 0.005)
                     # Balanced: not too high (0.05 = variance explosion) or too low
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network architecture MUST match Stage 0 for weight loading
POLICY_KWARGS = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    activation_fn=nn.ReLU
)

# Stage A specific
TOTAL_TIMESTEPS = 5_000_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 20
CHECKPOINT_FREQ = 250_000

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
    print("🔵 STAGE A: FOUNDATION - 7x7 WIDE CORRIDORS")
    print("="*70)
    print("Goal: Learn steering, obstacle avoidance, basic turns")
    print("Target: ≥60% success rate (5M steps)")
    print("\n🎯 ATTEMPT 14: Larger Cell Size + Stage 0 Transfer")
    print("   - BREAKTHROUGH: Increased cell_size (0.6m → 0.8m) = 80% diagnostic success!")
    print("   - Loading weights from Stage 0 (empty room expertise)")
    print("   - Same maze structure, more maneuvering space")
    print("   - heading_weight=0.005, time_penalty=-0.01, entropy_coef=0.01")
    print("="*70 + "\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_maze_dir = os.path.join(project_root, "assets", "train", "stage_a_foundation")
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_a_foundation")
    models_dir = os.path.join(project_root, "models", "stage_a")
    logs_dir = os.path.join(project_root, "logs", "stage_a")
    
    os.makedirs(models_dir, exist_ok=True)

    train_mazes = load_mazes(train_maze_dir)
    eval_mazes = load_mazes(eval_maze_dir)
    
    print(f"Loaded {len(train_mazes)} training mazes (7x7, wide)")
    print(f"Loaded {len(eval_mazes)} evaluation mazes (7x7, wide)\n")

    device = "cpu"

    train_env = SubprocVecEnv([
        make_env(train_mazes, render_mode="direct", terminate_on_collision=True)
        for _ in range(NUM_ENVS)
    ])
    eval_env = DummyVecEnv([
        make_env(eval_mazes, render_mode="direct", terminate_on_collision=True)
    ])

    checkpoint = os.path.join(models_dir, "best_model.zip")
    stage_0_model = os.path.join(project_root, "models", "stage_0", "best_model.zip")
    
    # ATTEMPT #14: Cell Size Breakthrough + Stage 0 Transfer
    # Cell size increased (0.6m → 0.8m): Same corridor structure, more maneuvering space
    # 80% diagnostic success with larger cell_size (vs 40% with 0.6m)
    # Load from Stage 0: Transfer basic navigation skills to wider maze
    # Target: ≥60% eval success (likely achievable given 80% diagnostic)

    if os.path.exists(checkpoint):
        print(f"Resuming Stage A from checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=train_env, tensorboard_log=logs_dir, device=device)
        remaining_timesteps = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"Completed: {model.num_timesteps:,} | Remaining: {remaining_timesteps:,}\n")
    elif os.path.exists(stage_0_model):
        print("\U0001f680 ATTEMPT #14: Cell Size Breakthrough + Stage 0 Transfer")
        print(f"   Source: {stage_0_model}")
        print("   Cell size: 0.6m → 0.8m (80% diagnostic success!)")
        print("   Loading Stage 0 weights: basic navigation + collision avoidance")
        print("   Target: ≥60% eval success rate\n")
        
        # Load weights from Stage 0 (empty room mastery)
        model = PPO.load(stage_0_model, env=train_env, tensorboard_log=logs_dir, device=device)
        
        # Reset timestep counter for fresh Stage A training
        model.num_timesteps = 0
        remaining_timesteps = TOTAL_TIMESTEPS
        print("✅ Loaded Stage 0 weights! Starting Stage A training\n")
    else:
        print("⚠️ WARNING: Stage 0 model not found! Training from scratch")
        print("   (Curriculum learning recommended: train Stage 0 first)\n")
        model = PPO(
            POLICY, train_env,
            policy_kwargs=POLICY_KWARGS,
            learning_rate=LEARNING_RATE,
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
        remaining_timesteps = TOTAL_TIMESTEPS
        print("✅ Model initialized! Distance-scaled heading ready\n")
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=models_dir,
        name_prefix="stage_a_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    gc_callback = GarbageCollectionCallback(gc_freq=10000)
    
    print("Starting Stage A training...")
    print(f"Monitor: tensorboard --logdir {logs_dir}\n")
    
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[eval_callback, checkpoint_callback, gc_callback],
        progress_bar=True
    )
    
    final_model_path = os.path.join(models_dir, "stage_a_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("✅ STAGE A COMPLETE!")
    print("="*70)
    print(f"Best model: {os.path.join(models_dir, 'best_model.zip')}")
    print(f"Final model: {final_model_path}")
    print("\n📊 Check TensorBoard - if success rate ≥60%, proceed to Stage B:")
    print("   python training/train_stage_b.py")
    print("="*70 + "\n")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
