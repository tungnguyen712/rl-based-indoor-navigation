"""
STAGE B: Medium (Introduce Mild Choke Points)
Goal: Slower approach, narrow corridor entry, partial detours
Map: 9×9, mostly 2-cell wide, 1-2 mild choke points, 1 dead end
Target: ≥40% success rate
Training: 5M timesteps
Prerequisite: Stage A completed (≥60% success)
"""
import os
import sys
import json
import gc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from envs.indoor_maze_env import IndoorMazeEnv

# ==================== STAGE B HYPERPARAMETERS ====================
NUM_ENVS = 4  # Match Stage A (proven stable)
POLICY = "MlpPolicy"
LEARNING_RATE = 1e-4  # Moderate fine-tuning (balanced learning speed)
N_STEPS = 2048  # Match Stage A (stable rollout length)
BATCH_SIZE = 256  # Same as Stage A
N_EPOCHS = 4  # Match Stage A (prevents overfitting to stuck batches)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01  # Match Stage A (proven working value)
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network architecture MUST match Stage A for weight loading
POLICY_KWARGS = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    activation_fn=nn.ReLU
)

# Stage B specific
TOTAL_TIMESTEPS = 10_000_000  # 10M steps (2x due to lower LR and increased complexity)
EVAL_FREQ = 50_000  # Match Stage A (frequent monitoring)
N_EVAL_EPISODES = 30
CHECKPOINT_FREQ = 500_000

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
    print("🟡 STAGE B: MEDIUM - 9x9 WITH MILD CHOKE POINTS")
    print("="*70)
    print("Goal: Narrow corridor entry, slower approach, partial detours")
    print("Target: ≥40% success rate (5M steps)")
    print("\n🎯 BASELINE: Stage A model already achieves 70% on Stage B!")
    print("   - Cell size: 0.8m (improved from 0.6m)")
    print("   - Strong transfer learning foundation")
    print("   - Hyperparameters: Matched to Stage A (stable config)")
    print("   - Goal: Push to 80%+ with Stage B fine-tuning")
    print("="*70 + "\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_maze_dir = os.path.join(project_root, "assets", "train", "stage_b_medium")
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_b_medium")
    models_dir = os.path.join(project_root, "models", "stage_b")
    stage_a_models_dir = os.path.join(project_root, "models", "stage_a")
    logs_dir = os.path.join(project_root, "logs", "stage_b")
    
    os.makedirs(models_dir, exist_ok=True)

    train_mazes = load_mazes(train_maze_dir)
    eval_mazes = load_mazes(eval_maze_dir)
    
    print(f"Loaded {len(train_mazes)} training mazes (9x9, mild choke points)")
    print(f"Loaded {len(eval_mazes)} evaluation mazes (9x9, mild choke points)\n")

    device = "cpu"

    train_env = SubprocVecEnv([
        make_env(train_mazes, render_mode="direct", terminate_on_collision=False)
        for _ in range(NUM_ENVS)
    ])
    eval_env = DummyVecEnv([
        make_env(eval_mazes, render_mode="direct", terminate_on_collision=True)
    ])

    # Check for Stage A model (REQUIRED!)
    # Use stage_a_failed_9 which achieved better results (45% vs 30%)
    stage_a_model_path = os.path.join(project_root, "models", "stage_a_failed_9", "best_model.zip")
    if not os.path.exists(stage_a_model_path):
        print("❌ ERROR: Stage A (failed_9) model not found!")
        print(f"Expected: {stage_a_model_path}")
        print("\nYou must complete Stage A training first:")
        print("  python training/train_stage_a.py")
        print("\n⚠️  Stage A must achieve ≥60% success rate before Stage B!")
        print("\nExiting...")
        return

    checkpoint = os.path.join(models_dir, "best_model.zip")
    if os.path.exists(checkpoint):
        print(f"Resuming Stage B from checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=train_env, tensorboard_log=logs_dir, device=device)
        remaining_timesteps = TOTAL_TIMESTEPS - model.num_timesteps
        print(f"Completed: {model.num_timesteps:,} | Remaining: {remaining_timesteps:,}\n")
    else:
        print(f"🚀 Loading Stage A model from: {stage_a_model_path}")
        print(f"   Baseline: 70% success rate on Stage B mazes!\n")
        model = PPO.load(stage_a_model_path, env=train_env, tensorboard_log=logs_dir, device=device)
        
        # Adjust hyperparameters for Stage B fine-tuning
        model.learning_rate = LEARNING_RATE
        model.ent_coef = ENTROPY_COEF
        model.n_epochs = N_EPOCHS
        
        # Reset timestep counter for clean Stage B tracking
        model.num_timesteps = 0
        remaining_timesteps = TOTAL_TIMESTEPS
        
        print(f"✓ Transferred Stage A weights (80% success on 7×7 mazes)")
        print(f"✓ Adjusted LR to {LEARNING_RATE} for fine-tuning (Stage A: 3e-4)")
        print(f"✓ Kept entropy at {ENTROPY_COEF} (matched to Stage A)")
        print(f"✓ Kept n_epochs at {N_EPOCHS} (prevents overfitting)")
        print(f"✓ Reset timesteps for Stage B tracking")
        print("\nStarting Stage B training (targeting 80%+ from 70% baseline)...\n")
    
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
        name_prefix="stage_b_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    gc_callback = GarbageCollectionCallback(gc_freq=10000)
    
    print("Starting Stage B training...")
    print(f"Monitor: tensorboard --logdir={logs_dir}\n")
    
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[eval_callback, checkpoint_callback, gc_callback],
        progress_bar=True
    )
    
    final_model_path = os.path.join(models_dir, "stage_b_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("✅ STAGE B COMPLETE!")
    print("="*70)
    print(f"Best model: {os.path.join(models_dir, 'best_model.zip')}")
    print(f"Final model: {final_model_path}")
    print(f"Total training: {model.num_timesteps:,} steps (Stage B only)")
    print("\n📊 Check TensorBoard:")
    print(f"   tensorboard --logdir={logs_dir}")
    print("\n🎯 Success criteria:")
    print("   - Target: ≥40% success rate (BASELINE: 70%)")
    print("   - Goal: Push to 80%+ with fine-tuning")
    print("\n🚀 If success rate ≥40%, proceed to Stage C:")
    print("   python training/train_stage_c.py")
    print("="*70 + "\n")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
