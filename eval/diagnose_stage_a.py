import os
import sys
import json
import time
import numpy as np
import pybullet as p

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from envs.indoor_maze_env import IndoorMazeEnv

# Create video output directory
video_output_dir = os.path.join(project_root, "videos", "7x7_foundation_mazes")
os.makedirs(video_output_dir, exist_ok=True)

def load_mazes(folder_path):
    maze_layouts = []
    for f in os.listdir(folder_path):
        if f.endswith(".json"):
            path = os.path.join(folder_path, f)
            with open(path, "r") as fp:
                maze_layouts.append(json.load(fp))
    return maze_layouts

def main():
    # Load model and mazes
    eval_maze_dir = os.path.join(project_root, "assets", "eval", "stage_a_foundation")
    model_path = os.path.join(project_root, "models", "stage_a", "best_model.zip")
    
    eval_mazes = load_mazes(eval_maze_dir)
    model = PPO.load(model_path, device="cpu")
    env = IndoorMazeEnv(maze_layouts=eval_mazes, render_mode="gui", terminate_on_collision=True)
    
    # Configure better video resolution
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    successes = 0
    failures = 0
    timeouts = 0
    
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        goal_pos = env.goal_pos
        
        # Start recording video
        video_filename = os.path.join(video_output_dir, f"episode_{ep+1}.mp4")
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)
        
        start_pos = env.start_pos
        goal_pos = env.goal_pos
        total_reward = 0.0
        step = 0
        done = False
        
        while not done and step < env.max_episode_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            done = terminated or truncated
        
        # Wait 1 second before stopping to capture final state
        time.sleep(1.0)
        
        # Stop recording video
        p.stopStateLogging(log_id)
        
        # Final outcome
        success = info.get("is_success", False)
        final_pos, _ = p.getBasePositionAndOrientation(env.robot_id)
        final_dist = np.linalg.norm(np.array([final_pos[0], final_pos[1]]) - np.array(goal_pos))
        
        if success:
            print(f"SUCCESS in {step} steps")
            successes += 1
        elif truncated:
            print(f"TIMEOUT at {step} steps (dist: {final_dist:.2f}m)")
            timeouts += 1
            # Delete video for failed episodes
            if os.path.exists(video_filename):
                os.remove(video_filename)
        else:
            print(f"FAILURE at {step} steps (dist: {final_dist:.2f}m)")
            failures += 1
            # Delete video for failed episodes
            if os.path.exists(video_filename):
                os.remove(video_filename)
        
        print(f"Total reward: {total_reward:.1f}")
        print(f"Final distance: {final_dist:.2f}m")
        
        input("\nPress ENTER for next episode...")

    print("SUMMARY")
    print(f"Successes: {successes}/10 ({successes*10}%)")
    print(f"Timeouts:  {timeouts}/10 ({timeouts*10}%)")
    print(f"Failures:  {failures}/10 ({failures*10}%)")
    
    env.close()

if __name__ == "__main__":
    main()
