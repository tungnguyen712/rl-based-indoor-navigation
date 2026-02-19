"""
STAGE C: Hard (True Maze with Precision Required)
Goal: Precision, loop-breaking, real navigation, multiple choke points
Map: 9×9, many 1-cell corridors, multiple choke points, 2-3 dead ends
Target: ≥25-30% success rate (challenging!)
Training: 10M timesteps
Prerequisite: Stage B completed (≥40% success)
"""
import os
import sys
import json
import gc

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

from envs.indoor_maze_env import IndoorMazeEnv

# ==================== STAGE C HYPERPARAMETERS ====================
NUM_ENVS = 8
POLICY = "MlpPolicy"
LEARNING_RATE = 5e-5  # very low for precision fine-tuning
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.02  # low entropy, exploit learned skills
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Stage C specific
TOTAL_TIMESTEPS = 10_000_000  # 10M steps
EVAL_FREQ = 100_000
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
    print("🔴 STAGE C: HARD - TRUE MAZE NAVIGATION")
    print("="*70)
    print("Goal: Precision, multiple choke points, dead-end handling")
    print("Target: ≥25-30% success rate (10M steps)")
    print("⚠️  This is HARD - 25-30% is respectable for PPO!")
    print("="*70 + "\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_maze_dir = os.path.join(project_root, "assets", "train", "stage_c_hard")
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_c_hard")
    models_dir = os.path.join(project_root, "models", "stage_c")
    stage_b_models_dir = os.path.join(project_root, "models", "stage_b")
    logs_dir = os.path.join(project_root, "logs")
    
    os.makedirs(models_dir, exist_ok=True)

    train_mazes = load_mazes(train_maze_dir)
    eval_mazes = load_mazes(eval_maze_dir)
    
    print(f"Loaded {len(train_mazes)} training mazes (9x9, many choke points)")
    print(f"Loaded {len(eval_mazes)} evaluation mazes (9x9, many choke points)\n")

    device = "cpu"

    train_env = SubprocVecEnv([
        make_env(train_mazes, render_mode="direct", terminate_on_collision=False)
        for _ in range(NUM_ENVS)
    ])
    eval_env = DummyVecEnv([
        make_env(eval_mazes, render_mode="direct", terminate_on_collision=True)
    ])

    # Check for Stage B model (REQUIRED!)
    stage_b_model_path = os.path.join(stage_b_models_dir, "best_model.zip")
    if not os.path.exists(stage_b_model_path):
        print("❌ ERROR: Stage B model not found!")
        print(f"Expected: {stage_b_model_path}")
        print("\nYou must complete Stage B training first:")
        print("  python training/train_stage_b.py")
        print("\n⚠️  Stage B must achieve ≥40% success rate before Stage C!")
        print("\nExiting...")
        return

    checkpoint = os.path.join(models_dir, "best_model.zip")
    if os.path.exists(checkpoint):
        print(f"Resuming Stage C from checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=train_env, tensorboard_log=logs_dir, device=device)
        stage_b_end = 8_000_000  # Stage A (3M) + Stage B (5M)
        stage_c_completed = max(0, model.num_timesteps - stage_b_end)
        remaining_timesteps = TOTAL_TIMESTEPS - stage_c_completed
        print(f"Stage C completed: {stage_c_completed:,} | Remaining: {remaining_timesteps:,}\n")
    else:
        print(f"✓ Loading Stage B model from: {stage_b_model_path}")
        model = PPO.load(stage_b_model_path, env=train_env, tensorboard_log=logs_dir, device=device)
        
        model.learning_rate = LEARNING_RATE
        model.ent_coef = ENTROPY_COEF
        
        print(f"✓ Transferred knowledge from Stage B ({model.num_timesteps:,} steps)")
        print(f"✓ Adjusted LR to {LEARNING_RATE} for precision fine-tuning")
        print(f"✓ Adjusted entropy to {ENTROPY_COEF} (exploit skills)")
        print("\nStarting Stage C - the final challenge...\n")
        remaining_timesteps = TOTAL_TIMESTEPS
    
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
        name_prefix="stage_c_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    gc_callback = GarbageCollectionCallback(gc_freq=10000)
    
    print("Starting Stage C training (this will take a while)...")
    print(f"Monitor: tensorboard --logdir {logs_dir}\n")
    
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=[eval_callback, checkpoint_callback, gc_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    final_model_path = os.path.join(models_dir, "stage_c_final.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("🎉 STAGE C COMPLETE - TRAINING FINISHED!")
    print("="*70)
    print(f"Best model: {os.path.join(models_dir, 'best_model.zip')}")
    print(f"Final model: {final_model_path}")
    print(f"Total training: {model.num_timesteps:,} steps")
    print("\n🏆 Your robot has completed the full curriculum!")
    print("   If you achieved 25-30% on Stage C, that's excellent for PPO!")
    print("\n📊 Evaluate final performance:")
    print("   python training/evaluate.py")
    print("="*70 + "\n")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
