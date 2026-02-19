# RL Indoor Navigation - Context Summary

**Date:** February 16, 2026 (Updated - Late Evening Session)  
**Status:** Stage A Attempt 7 fixes applied — ready for retraining  
**Current Challenge:** Retrain Stage A with 4 critical fixes applied

---

## CRITICAL LESSON: REWARD HACKING **AND REWARD SPARSITY** ARE THE MAIN ENEMIES

Throughout this project, there have been two primary failure modes:
1. **Reward hacking:** Agents exploiting reward structures rather than learning intended behavior
2. **Reward sparsity:** Harsh penalties for necessary intermediate actions create learning valleys

**Latest Discovery:** In mazes, robots must temporarily move AWAY from goals to navigate around obstacles. The symmetric distance reward (`+100 × delta` for both directions) punishes this necessary maneuvering as harshly as it rewards progress, causing the agent to prefer crashing over turning.

---

## Problem History - What Failed & Why

### Phase 1: Ackermann Steering (FAILED - 10% success)
- Robot couldn't turn in place → stuck in corners
- Forward-only movement incompatible with maze navigation
- Minimum turning radius too large for tight spaces
- **Key insight:** Architecture matters more than tuning

### Phase 2: Complex Reward Structures (FAILED - reward hacking)
- Multiple reward components created conflicting signals
- Step penalties accumulated larger than any positive reward
- Agent learned to stand still or hover to avoid penalties
- **Key insight:** Simplicity prevents exploitation

### Phase 3: Pure Distance Shaping (FAILED - 60% success)
- Robot oscillated/jittered near goal without finishing
- No directional guidance → random walk behavior
- Agent couldn't learn which way to turn from distance alone
- **Key insight:** Some guidance is necessary

### Phase 4: Heading + Proximity Bonuses (FAILED - 37% success w/ 500+ reward)
- **CRITICAL FAILURE - REWARD HACKING:**
  - heading_weight=0.3-0.5: +0.3-0.5/step × 1000 steps = 300-500 reward
  - proximity_bonus=0.5-1.0: +0.5-1.0/step × 1000 steps = 500-1000 reward
  - **TOTAL hovering reward: 800-1500** vs **goal_reward: 200-500**
- Agent learned: "Stand still near goal collecting bonuses forever"
- Mean reward 500+ but only 37% success = clear exploitation
- **Key insight:** Bonus rewards must be MUCH smaller than goal reward

---

## Current Solution (FINALIZED & PROVEN)

### Architecture Changes ✅

#### 1. **Differential Drive Robot**
```python
# File: envs/indoor_maze_env.py
action_space = Box(low=[-1.0, -1.0], high=[1.0, 1.0])  # [left_wheel, right_wheel]

# Benefits:
# - Can turn in place (no minimum turning radius)
# - Full 360° maneuverability
# - Steering joints locked at 0
```

#### 2. **Enhanced Observations (71 dimensions)**
```python
observation = [
    lidar_64_rays,        # 64 dims - obstacle detection
    goal_distance,        # 1 dim - normalized [0,1]
    goal_angle,          # 1 dim - CRITICAL for learning which way to turn
    velocity_x,          # 1 dim - linear vel x
    velocity_y,          # 1 dim - linear vel y  
    angular_velocity_z,  # 1 dim - rotation rate
    prev_action_left,    # 1 dim - left wheel from last step
    prev_action_right    # 1 dim - right wheel from last step
]  # Total: 71 dims
```

**Why goal_angle matters:** Agent can learn heading from observations, but reward provides strong guidance signal during training.

#### 3. **Environment Parameters**
```python
max_episode_steps = 1000      # CRITICAL: Allow full 360° turn from any spawn
goal_radius = 1.5             # Generous for empty room (3×3m space)
physics_steps_per_action = 12 # Responsive control
max_wheel_speed = 10.0        # rad/sec for differential drive
```

---

## Reward Structure (FINAL - MATHEMATICALLY PROVEN)

### The Math That Prevents Reward Hacking

```python
# File: envs/indoor_maze_env.py
self.goal_reward = 500.0      # Dominates episode (anti-hovering)
self.dist_weight = 100.0      # Primary signal: movement toward goal
self.heading_weight = 0.05    # Gentle guidance (NON-exploitable)
```

**Proof hovering is unprofitable:**
```
Strategy A: Hover aligned at goal (1000 steps):
  - Heading bonus: 0.05 × 1000 = 50
  - Distance progress: 0
  - Goal bonus: 0
  TOTAL: 50 reward

Strategy B: Reach goal (300 steps):
  - Distance progress: 100 × 3 meters = 300
  - Heading bonus: 0.05 × 300 = 15
  - Goal bonus: 500
  TOTAL: 815 reward

Success is 16.3× more profitable than hovering ✓
```

### Implementation
```python
def compute_reward(self):
    reward = 0.0
    
    if self.prev_goal_dist is not None:
        # PRIMARY: Distance shaping (+100/m toward, -100/m away)
        dist_delta = self.prev_goal_dist - goal_dist
        reward = self.dist_weight * dist_delta
        
        # SECONDARY: Heading alignment (gentle guidance)
        angle_diff = goal_angle - robot_yaw
        angle_diff = arctan2(sin(angle_diff), cos(angle_diff))  # [-π, π]
        heading_alignment = cos(angle_diff)  # 1=aligned, -1=opposite, 0=perp
        reward += self.heading_weight * heading_alignment  # ±0.05/step
    
    # DOMINANT: Success bonus
    if succeeded:
        reward += self.goal_reward  # +500
    
    return reward
```

### Why These Specific Numbers?

**goal_reward=500 (not 200, not 2000):**
- 200 was too small → hovering competitive
- 2000 too large → value function instability (huge variance)
- 500 is 2×typical episode reward → healthy RL practice

**dist_weight=10.0 (reduced from 100):**
- **CRITICAL FIX:** 100 caused "jittering/wiggling" because moving 1cm (+1.0) was 10× more rewarding than turning (+0.1).
- At 10.0, moving 1cm (+0.1) ≈ turning (+0.1).
- Agent is no longer addicted to rushing blindly; it pauses to align.

**heading_weight=0.1 (increased from 0.05):**
- 0.05 was too weak, leading to oscillation/timeouts
- 0.1 provides stronger guidance (1000 steps = 100 reward < 500 goal)
- Added **backward driving penalty** (-0.05) because robot has 180° FOV (blind behind)

---

## Training Configuration

### Hyperparameters (train_stage_0.py)
```python
NUM_ENVS = 8              # Reduced from 16 (fixes CPU thrashing 95%→20% spikes)
LEARNING_RATE = 3e-4      # Standard PPO
N_STEPS = 2048
BATCH_SIZE = 256          # Large for stable gradients
N_EPOCHS = 4              # Prevents overfitting
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.05       # Moderate (with heading reward, needs less exploration)
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

POLICY_KWARGS = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128])  # Balanced (not 256×256)
)

TOTAL_TIMESTEPS = 2_000_000
USE_LR_SCHEDULE = True    # Decay for late-stage stability
```

### Stage 0: Empty Room Mazes
- 3 training mazes (3×3m rooms, different cell sizes)
- 1 evaluation maze
- No obstacles except boundary walls
- **Target:** ≥90% success before proceeding to Stage A

---

## Diagnostic Results - Proof of Concept

### GOOD: With heading_weight=0.05 (diagnostic at 900 steps, earlier session)
```
Episode 1: START facing away (← -0.86), dist=1.41m
  Step 550: Turned around (→ 0.97), dist=2.09m
  Step 600: Moving toward goal, dist=1.50m
  Result: ✅ SUCCESS in 644 steps

Episode 5: START 3.0m away
  Spent time aligning and navigating longer distance
  Result: ✅ SUCCESS in 806 steps

Overall: 100% success (10/10) with 900 steps
```
**Conclusion:** Heading reward enables smooth navigation.

### BAD: Without heading reward (pure distance shaping)
```
Episode 1: Oscillating at 1.43-1.49m for 1000 steps
  Facing away (← -0.65) entire episode
  Jittering back and forth with no progress
  Result: ⏱️ TIMEOUT

Episode 8: Started 1.00m away, moved AWAY to 1.41m
  Random walk behavior, no sense of direction
  Result: ⏱️ TIMEOUT

Overall: 60% success, mostly from lucky spawns (<1m distance)
```
**Conclusion:** Pure distance shaping insufficient for navigation.

### BAD: With too-large heading_weight=0.3-0.5
```
TensorBoard showed:
  - Mean reward: 500+ (very high)
  - Success rate: 37% (very low)
  - Episode length: 700-800 (near timeout)

Diagnostic showed:
  - Robot hovering 1.2-2.0m from goal
  - Not making progress toward finish
  - Collecting heading bonuses indefinitely

Result: REWARD HACKING confirmed
```
**Conclusion:** Large bonuses enable exploitation.

---

## Known Issues & Solutions ✓

| Issue | Cause | Solution | Status |
|-------|-------|----------|--------|
| CPU thrashing 95%→20% | NUM_ENVS=16 too many processes | Reduced to 8 | ✓ Fixed |
| Robot standing still | Heading/proximity bonuses too large | Reduced heading to 0.05, removed proximity | ✓ Fixed |
| Robot oscillating near goal | No directional gradient | Added heading_weight=0.05 | ✓ Fixed |
| Episode timeout before turning | max_steps=400-500 too short | Increased to 1000 | ✓ Fixed |
| Agent learning to hover | goal_reward too small vs bonuses | Increased to 500 (16:1 ratio) | ✓ Fixed |

---

## Expected Training Results

### Target Metrics (Stage 0 - Empty Room)
- **Success rate:** 85-95% (stable, minimal oscillation)
- **Mean episode length:** 300-500 steps (efficient paths)
- **Mean reward:** 700-850 (dominated by goal bonus + distance)
- **Training time:** ~60-90 minutes for 2M timesteps

### Red Flags to Watch For
| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Success <70% at 1M steps | Reward still broken | Review diagnostic, check for new exploitation |
| Episode length >700 | Robot wandering, not learning | Increase heading_weight slightly (0.05→0.08) |
| Reward >500 but success <40% | REWARD HACKING | Reduce bonus weights, increase goal reward |
| CPU 20-95% spikes | Too many parallel envs | Reduce NUM_ENVS further (8→6 or 4) |

---

## Commands Reference

```powershell
# Clean retrain Stage 0 (RECOMMENDED NEXT STEP)
Remove-Item -Recurse -Force models/stage_0, logs/stage_0
python training/train_stage_0.py

# Monitor training (in separate terminal)
tensorboard --logdir=logs

# CRITICAL: Run diagnostic to see actual behavior
python diagnose_stage_0.py

# Evaluate after training (optional)
python training/evaluate.py
```

---

## Critical Rules for Next Session

### ✅ DO:
1. **Keep reward ratios:** goal=500, dist=100, heading=0.05 (proven math)
2. **Run diagnostic first:** Before changing anything, see what robot does
3. **Trust heading reward:** Agent needs directional guidance (proven by A/B test)
4. **Monitor for exploitation:** High reward + low success = reward hacking
5. **Give enough time:** 1000 steps for worst-case 180° turn + navigation

### ❌ DON'T:
1. **Remove heading reward:** Robot can't navigate without it (60% vs 100% success)
2. **Increase heading above 0.1:** Enables hovering exploitation
3. **Add proximity bonuses:** Always leads to "hover near goal" strategy
4. **Increase goal above 1000:** Creates value function instability
5. **Reduce episode steps below 800:** Robot needs time to turn around

---

## Next Steps After Stage 0 Success (≥90%)

### Stage A: Wide Corridors (7×7, 0.6m cells)
**Changes:**
- **Keep reward identical:** goal=500, dist=100, heading=0.05 (it generalizes!)
- Increase max_episode_steps → 1500 (more complex paths)
- Target: 60%+ success rate
- **DO NOT touch reward ratios**

### If Stage A Fails (<40% success):
1. Check if mazes are solvable (run diagnostic)
2. Consider easier intermediate stage (5×5 with single wall)
3. Increase max_episode_steps to 2000
4. **Still don't change reward ratios - they work**

---

## Files Modified (Current State)

### Core Environment
**`envs/indoor_maze_env.py`:**
- Differential drive: [left_wheel, right_wheel]
- 71-dim observations: lidar + goal + velocity + prev_action
- Reward: goal=500, dist=100, heading=0.05 (anti-hacking ratios)
- max_episode_steps=1000, goal_radius=1.5m
- physics_steps_per_action=12

### Training
**`training/train_stage_0.py`:**
- NUM_ENVS=8, ENTROPY_COEF=0.05
- Network: 128×128 (balanced size)
- Learning rate schedule enabled
- 2M timesteps

### Diagnostic
**`diagnose_stage_0.py`:**
- Shows heading with symbols: → ← ↻
- Tracks progress every 50 steps
- Reveals exploitation patterns

---

## Git Recommendation

```powershell
git add .
git commit -m "Stage 0: Final reward structure - goal=500, dist=100, heading=0.05

- Mathematically proven anti-reward-hacking ratios (16:1 success vs hovering)
- max_episode_steps=1000 for full rotation capability
- NUM_ENVS=8 to prevent CPU thrashing
- Heading reward (0.05) provides directional guidance without exploitation
- Ready for final training run"
```

---

## Stage A: Attempt 6 - Distance-Scaled Heading (FAILED - 10% success)

**Date:** Feb 16, 2026 (Evening Session)  
**Config:**
- Fixed heading scale to use constant fade radius (1.0m) instead of map-dependent
- Restored Stage 0 proven scale: `goal_reward=500`, `dist_weight=100`, `heading_weight=0.05`
- Increased collision penalty to `-20.0` (was `-2.0`)
- Transfer learning from Stage 0

**Result:** 10% success, 90% early crashes (<100 steps typically)

**Symptoms:**
```
Episode 3: CRASH at step 38 (only moved 0.01m)
Episode 4: CRASH at step 99 (only moved 0.28m)
Episode 9: CRASH at step 77 (moved 0.67m but crashed)
```

**Root Cause Identified:** **ASYMMETRIC NAVIGATION PENALTY**
- In mazes, robots MUST move away from goal temporarily to navigate around walls
- Current: `reward = 100.0 * (prev_dist - curr_dist)`
- When robot turns to avoid wall: `curr_dist > prev_dist` → `reward = 100.0 * (-0.01) = -1.0`
- **This punishes necessary maneuvering as harshly as it rewards progress**
- Robot learns: "Turning hurts (-1.0 per step). Crashing hurts (-20.0 once). Just drive straight!"

**The Math:**
```
Strategy A: Maneuver around wall (5 steps backward, then 10 steps forward)
  - Backward: 5 × (-1.0) = -5.0
  - Forward: 10 × (+1.0) = +10.0
  - Total: +5.0
  
Strategy B: Drive straight into wall
  - Forward: 3 × (+0.5) = +1.5
  - Collision: -20.0
  - Total: -18.5

Strategy A is better, but the -5.0 penalty during maneuvering creates SPARSE reward.
Agent can't "see" the +10.0 payoff through the -5.0 valley.
```

**Solution:** **Asymmetric Distance Reward**
```python
dist_delta = self.prev_goal_dist - goal_dist
if dist_delta >= 0:
    # Moving closer: Full reward
    reward = self.dist_weight * dist_delta
else:
    # Moving away (necessary maneuvering): Gentle penalty
    reward = self.dist_weight * dist_delta * 0.1  # or 0.2
```

This creates:
- Progress reward: +100 per meter closer
- Maneuver penalty: -10 per meter away (10× more lenient)
- Collision penalty: -20 (keeps safety)

**Files Modified (Session 6):**
- `envs/indoor_maze_env.py`: 
  - Fixed heading scale logic (line ~503): `distance_factor = np.clip(goal_dist / 1.0, 0.0, 1.0)`
  - Restored reward scale: `goal_reward=500`, `dist_weight=100`, `collision_penalty=-20`
  - **STILL NEEDS:** Asymmetric distance penalty

**Files Status:**
- `models/stage_a_failed_run5/` - Attempt 6 checkpoint (10% success)
- `logs/stage_a/PPO_1/` - Attempt 6 logs (needs cleanup or rename)
- `models/stage_0/best_model.zip` - Proven baseline (>80% empty room)

---

**NEXT SESSION ACTION PLAN:**

### ✅ Critical Fix Required (15 min)
1. **Apply Asymmetric Distance Reward** in `envs/indoor_maze_env.py`:
   ```python
   # In compute_reward(), replace:
   dist_delta = self.prev_goal_dist - goal_dist
   reward = self.dist_weight * dist_delta
   
   # With:
   dist_delta = self.prev_goal_dist - goal_dist
   if dist_delta >= 0:
       reward = self.dist_weight * dist_delta  # Full reward for progress
   else:
       reward = self.dist_weight * dist_delta * 0.15  # 85% forgiveness for maneuvering
   ```

2. **Optional Tuning:**
   - Consider reducing `dist_weight` from 100 to 50 (less aggressive, smoother gradients)
   - Keep `collision_penalty = -20` (proven to stop profitable crashing)

3. **Clean Workspace:**
   ```powershell
   Remove-Item -Recurse -Force models/stage_a
   Remove-Item -Recurse -Force logs/stage_a/PPO_1
   ```

4. **Retrain Stage A:**
   ```powershell
   python training/train_stage_a.py
   ```
   - Will load from `models/stage_0/best_model.zip`
   - Target: 40-60% success after 2-3M steps
   - Monitor TensorBoard for success rate trend

---

## Stage A Attempt 7: Comprehensive Fix (4 Root Causes Identified)

**Date:** Feb 16, 2026 (Late Evening)  
**Status:** Fixes applied, ready for retraining

### 🔍 Root Cause Analysis (TensorBoard + Diagnostic)

**TensorBoard Evidence (3 runs compared):**
- `stage_a\PPO_1` (Attempt 6, purple): 10% success, value_loss=172, train/std=6
- `stage_a_failed_3` (pink): 20-33% success, train/std=50
- `stage_a_failed_5` (orange): std exploding to 135

**4 Critical Bugs Found:**

#### Bug 1: REWARD SCALE MISMATCH (Critical)
- Stage 0 model trained with `goal_reward=10, dist_weight=1.0` → mean rewards ~10
- Attempt 6 env had `goal_reward=500, dist_weight=100` → rewards in 500-800 range
- Value function from Stage 0 expected rewards ~10, received rewards 100× larger
- **TensorBoard proof:** `value_loss=172` (should be 5-20 for matched scale)
- **Fix:** Restored `goal_reward=10.0, dist_weight=1.0` to match Stage 0

#### Bug 2: BACKWARD DRIVING EXPLOIT (Critical)
- Robot drives BACKWARDS toward goal (heading consistently negative while distance decreases)
- 180° front-facing lidar means robot is BLIND behind it → crashes into walls
- Old backward penalty (-0.05/step) negligible vs distance reward (+1.0/step)
- **Diagnostic proof:** Episodes 2,4,7,9 show heading ←(-0.6 to -1.0) with decreasing distance
- **Fix:** Zero distance reward for backward progress + strong backward penalty (-0.5/step)

#### Bug 3: ENTROPY COLLAPSE (Moderate)
- Stage A used `ENTROPY_COEF=0.01` while Stage 0 used `0.05`
- Low entropy caused policy to collapse into backward-driving local minimum
- **TensorBoard proof:** `train/std=6` (vs failed_3 had std=50) → barely exploring
- **Fix:** Changed `ENTROPY_COEF=0.05` in train_stage_a.py

#### Bug 4: SUCCESS REWARD NEVER APPLIED (Critical)  
- In `step()`, `self.succeeded_this_step` was set AFTER `compute_reward()` was called
- `compute_reward()` checks `getattr(self, "succeeded_this_step", False)` → always `False`
- **The +10.0 goal reward was NEVER given in ANY training run (including Stage 0)**
- Stage 0 achieved 80%+ using only distance shaping (goal bonus = 0 due to bug)
- **Fix:** Moved `check_success()` BEFORE `compute_reward()` in `step()`

### ✅ All Fixes Applied

**`envs/indoor_maze_env.py` changes:**
```python
# Reward parameters (matched to Stage 0 scale)
self.goal_reward = 10.0       # NOW actually applied (bug fixed)
self.dist_weight = 1.0        # +1.0 per meter toward goal
self.heading_weight = 0.05    # Gentle guidance
self.collision_penalty = -5.0 # Wall collision
self.time_penalty = -0.001    # Existence tax
self.backward_penalty = -0.5  # NEW: Strong penalty for blind backward driving

# Reward function (compute_reward):
# 1. Distance: Forward-gated, asymmetric
#    - Forward + toward goal: FULL reward (dist_weight × delta)
#    - Backward + toward goal: ZERO reward (prevents blind navigation)
#    - Moving away from goal: 20% penalty (allows maneuvering)
# 2. Heading: Simple alignment (heading_weight × cos(angle_diff))
# 3. Backward penalty: -0.5 per step when forward_speed < -0.1
# 4. Success bonus: +10.0 (NOW applied correctly)
# 5. Collision: -5.0
# 6. Time: -0.001 per step

# step() method fix:
# BEFORE: reward = compute_reward() → check_success() → set succeeded_this_step
# AFTER:  check_success() → set succeeded_this_step → reward = compute_reward()
```

**`training/train_stage_a.py` changes:**
```python
ENTROPY_COEF = 0.05  # Matches Stage 0 (was 0.01, caused policy collapse)
```

### Expected Reward Profile (Attempt 7)
```
Successful forward episode (3m, 500 steps):
  Distance: +1.0 × 3m = +3.0
  Heading:  +0.05 × 0.5avg × 500 = +12.5
  Time:     -0.001 × 500 = -0.5
  Goal:     +10.0 (NOW applied!)
  Total:    ~+25.0

Backward-driving crash (200 steps):
  Distance: 0 (zeroed for backward motion)
  Heading:  0.05 × (-0.8) × 200 = -8.0
  Backward: -0.5 × 200 = -100.0
  Collision: -5.0
  Total:    -113.0 (STRONGLY unprofitable)
```

### Stage 0 Verification
```
Last 10 mean rewards: [9.61, 9.64, 9.68, 7.93, 10.27, 9.66, 11.35, 9.26, 10.15, 9.79]
num_timesteps: 1,924,692
ent_coef: 0.05
Network: pi=[128,128], vf=[128,128]
```

### TensorBoard Metrics to Watch
| Metric | Healthy | Red Flag |
|--------|---------|----------|
| value_loss | 5-30 | >50 (reward scale mismatch) |
| train/std | 15-60 | <10 (policy collapse) |
| eval/success_rate | Increasing | Flat at <20% after 1M steps |
| eval/mean_reward | 10-25 (success), -5 to 0 (fails) | >100 (scale mismatch) |

### Retraining Commands
```powershell
# Clean workspace
Remove-Item -Recurse -Force models/stage_a -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force logs/stage_a -ErrorAction SilentlyContinue

# Retrain (loads from models/stage_0/best_model.zip)
python training/train_stage_a.py

# Monitor
tensorboard --logdir=logs

# After training, diagnose
python diagnose_stage_a.py
```

### Files Modified (Current State - Post Attempt 7 Fixes)
- `envs/indoor_maze_env.py`: Reward function rewritten, step() bug fixed
- `training/train_stage_a.py`: ENTROPY_COEF = 0.05
- `models/stage_a_failed_run5/`: Attempt 6 checkpoint (10% success)
- `models/stage_0/best_model.zip`: Proven baseline (>80% empty room)

---

**END OF CONTEXT DOCUMENT**  
**Last Updated:** February 16, 2026 (Late Evening)  
**Status:** Stage 0 Complete (>80%), Stage A Attempt 7 Ready  
**Next Action:** Clean workspace, retrain Stage A with all 4 fixes applied

---

## Current Status (Feb 16, 2026)

### ✅ Stage 0: Empty Room (COMPLETE)
- **Outcome:** Success Rate >80% (diagnostic showed 100%).
- **Final Reward Structure:**
  ```python
  goal_reward = 10.0        # Success bonus (rescaled from 500)
  dist_weight = 1.0         # Distance shaping (rescaled from 100)
  heading_weight = 0.005    # Gentle guidance (rescaled from 0.05)
  collision_penalty = -1.0  # Wall hit penalty
  time_penalty = -0.001     # Existence tax
  ```
- **Critical Parameters:**
  - `goal_radius = 0.5m` (fixed "fake success")
  - `max_episode_steps = 1500` (allow full navigation)
  - `NUM_ENVS = 4` (prevent CPU thrashing)
  - `terminate_on_collision = False` (allow recovery)

---

## Stage A: Foundation - 7x7 Mazes (5 FAILED ATTEMPTS)

**Target:** ≥60% success rate on 7×7 mazes with wide corridors  
**Environment:** `terminate_on_collision=True`, 5M timesteps  
**Strategy:** Transfer learning from Stage 0 model

### 🔴 Attempt 1: SAC Algorithm (FAILED - 0% success)
**Date:** Feb 15, 2026  
**Config:**
- Algorithm: SAC (Soft Actor-Critic)
- Loaded from Stage 0 weights
- Original reward structure

**Result:** 0% success after 3.7M steps

**Root Cause:** SAC fundamentally incompatible with discrete maze navigation
- Continuous action optimization struggles with maze topology
- PPO's on-policy learning better suited for sparse rewards
- SAC's entropy regularization caused excessive random exploration

**Lesson:** Algorithm choice matters more than tuning. Stick with PPO.

---

### 🔴 Attempt 2: PPO + High Entropy (FAILED - 40% → 0% collapse)
**Date:** Feb 15, 2026  
**Config:**
- Algorithm: PPO with transfer learning from Stage 0
- `ENTROPY_COEF = 0.05` (moderate exploration)
- `heading_weight = 0.1` (stronger than Stage 0)
- `collision_penalty = -50.0` (very harsh)
- Loaded Stage 0 best_model

**Result:** Peaked at 40% success around 2.8M steps, collapsed to 0% by 5M steps

**Symptoms:**
- TensorBoard: Success rate crashed to flat 0%
- Diagnostic: Agent "staring at wall" - facing goal but not moving
- Episode 8: 1500 steps, moved only 0.07m, `heading=→(1.00)` entire time

**Root Cause:** "Staring at wall" reward hacking
- Heading reward (+0.1/step) paid out while standing still
- Agent learned: Face goal through wall, collect +150 heading reward over 1500 steps
- Collision penalty too harsh (-50) made agent risk-averse

**Lesson:** Heading reward must be conditional on movement OR removed entirely.

---

### 🔴 Attempt 3: PPO + Low Entropy (FAILED - 40% → 0% collapse)  
**Date:** Feb 15, 2026  
**Config:**
- `ENTROPY_COEF = 0.01` (very low, attempt to stabilize)
- `heading_weight = 0.005` (reduced to minimize exploit)
- `collision_penalty = -1.0` (reverted to reasonable)
- `time_penalty = -0.001`
- Added collision penalty to reward function
- Loaded from Stage 0

**Result:** Similar pattern - peaked ~40%, collapsed to 0%

**Symptoms:**
- TensorBoard: train/std exploded to 9.9M+ (catastrophic variance explosion)
- Policy completely destabilized
- Success rate dropped to 0.0062

**Root Cause:** Variance explosion due to transfer learning instability
- Low entropy prevented exploration recovery
- Policy log_std diverged without regulation
- Transfer learning + low entropy = unstable gradient updates

**Lesson:** Entropy too low prevents recovery from bad states. Need log_std reset when loading checkpoints.

---

### 🟡 Attempt 4: PPO + High Penalties (PARTIAL SUCCESS - 30-60% stable)
**Date:** Feb 16, 2026  
**Config:**
- `ENTROPY_COEF = 0.01`
- `heading_weight = 0.005`
- `collision_penalty = -5.0` (5× stronger)
- `time_penalty = -0.01` (10× stronger)
- Log_std reset on load
- Loaded from failed run 3's best checkpoint

**Result:** Maintained 30-60% success throughout training (BEST SO FAR)

**Symptoms:**
- TensorBoard: Stable variance (std ~104, no explosion)
- Success rate 40-60% (eval), 42% (rollout)
- Diagnostic: Still shows "staring" in Episode 8 (1500 steps, 0.13m progress)

**Analysis:**
- ✅ No variance explosion (log_std reset worked)
- ✅ Stable learning curve
- ❌ Still exploits heading reward (40% timeout rate)
- ❌ High penalties didn't eliminate exploit

**Lesson:** Run 4 is baseline. Heading reward itself is the problem, not penalty strength.

---

### 🔴 Attempt 5: PPO + Conditional Heading (FAILED - 18% → 5% collapse)
**Date:** Feb 16, 2026  
**Config:**
- Conditional heading: `if forward_speed > 0.1: reward += heading_alignment`
- `heading_weight = 0.01`
- `collision_penalty = -1.0` (reverted from Run 4)
- `time_penalty = -0.001` (reverted from Run 4)
- Loaded from Run 4's best model

**Result:** Collapsed from 18% to 5% success

**Symptoms:**
- TensorBoard: Worse than ALL previous runs
- Success dropped continuously
- Diagnostic: Episode 8 STILL staring at wall (moved 0.01m in 1500 steps)

**Root Causes:**
1. **Reward distribution shift:** Run 4 trained with penalties (-5.0, -0.01), Run 5 env has (-1.0, -0.001)
2. **Conditional logic failed:** Robot learned to barely wiggle (speed ~0.11) to pass threshold
3. **Complexity backfired:** Added logic didn't fix exploit, broke learning instead

**Lesson:** 
- Reward mismatch between checkpoint and environment kills training
- Conditional rewards are hard to tune correctly
- Simpler is better - remove heading entirely instead of making it conditional

---

## Stage A Summary: What We Learned

### Failed Approaches
| Attempt | Strategy | Peak Success | End Success | Why Failed |
|---------|----------|--------------|-------------|------------|
| 1 | SAC | 0% | 0% | Wrong algorithm |
| 2 | High entropy + harsh collision | 40% | 0% | Heading exploit + variance explosion |
| 3 | Low entropy + collision fix | 40% | 0% | Variance explosion, entropy too low |
| 4 | High penalties | **60%** | **60%** | ✅ Stable but still has exploit |
| 5 | Conditional heading | 18% | 5% | Reward mismatch + broken conditional |

### Critical Insights

**1. Heading Reward is Inherently Exploitable**
- Unconditional heading → "staring at wall" (Runs 2, 3, 4)
- Conditional heading → agent games threshold (Run 5)
- **Solution:** Remove heading reward entirely. Distance shaping implicitly rewards orientation.

**2. Transfer Learning Requires Reward Consistency**  
- Run 5 loaded Run 4's weights trained with (-5.0, -0.01) penalties
- Environment changed to (-1.0, -0.001) penalties
- Result: Catastrophic collapse
- **Solution:** Match environment rewards to checkpoint OR train from scratch

**3. Variance Explosion = Policy Death**
- Run 3: train/std → 9.9M+ (complete breakdown)
- Cause: Transfer learning + low entropy + no log_std reset
- **Solution:** Always reset log_std when loading checkpoints

**4. Penalties Don't Fix Exploits**
- Run 4 tried -5.0 collision, -0.01 time penalties
- Agent still preferred safe exploitation over risky navigation
- **Solution:** Fix reward structure, not penalty magnitude

---

## Next Action (Decision Point)

### Current Environment (Post-Run 5)
```python
# envs/indoor_maze_env.py
goal_reward = 10.0 
dist_weight = 1.0         # ONLY reward component
collision_penalty = -1.0
time_penalty = -0.001
heading_weight = 0.0      # REMOVED
```

### Options for Attempt 6

**Option A: Train from Scratch (RECOMMENDED)**
- Fresh start with pure distance shaping
- No reward mismatch issues
- Clean learning signal
- Pro: No baggage, Con: Slower (no transfer learning)

**Option B: Load Stage 0**
- Transfer from empty room baseline
- Unknown what reward structure Stage 0 used
- Pro: Some transfer benefit, Con: Reward uncertainty

**Option C: Accept Run 4 and Move to Stage B**
- Run 4 got 50% diagnostic success (close to 60% target)
- Stable training, no collapse
- Pro: Fastest, Con: Still has heading exploit

**Option D: Load Run 4, Keep Its Reward Structure**
- Revert environment to Run 4's config
- Keep heading reward but start Stage B
- Pro: Known stable, Con: Doesn't fix exploit

### Recommendation
**Train from scratch (Option A)** with pure distance shaping. This is the cleanest solution that:
- Eliminates heading exploit completely
- No reward mismatch issues
- Tests hypothesis that distance alone is sufficient
- If it works: Clean baseline for Stage B
- If it fails: Consider Option C (accept 50%)

---

## Commands for Next Session

### Option A: Train From Scratch (RECOMMENDED)
```powershell
# Remove failed run 4's checkpoint reference
# Training script will fall through to fresh initialization
python training/train_stage_a.py

# Monitor in separate terminal
tensorboard --logdir=logs

# After training, run diagnostic
python diagnose_stage_a.py
```

### Option B: Validate Pure Distance in Stage 0 First
```powershell
# Test if distance-only works in simpler environment
# Temporarily remove heading from Stage 0, retrain
python training/train_stage_0.py
python diagnose_stage_0.py
```

### Option C: Accept Run 4 and Move On
```powershell
# Restore Run 4 as current model
Move-Item models/stage_a_failed_run4/best_model.zip models/stage_a/best_model.zip

# Proceed to Stage B with known baseline
python training/train_stage_b.py
```

---

## Technical Debt / Action Items
- [ ] **CRITICAL:** Document Stage 0's actual reward structure (need version control)
- [ ] Update train_stage_a.py to train from scratch by removing prev_best_model fallback
- [ ] Run diagnostic on Stage 0 model to verify reward structure
- [ ] Consider git commit after each training run for rollback capability
- [ ] Add automated diagnostic after training (detect exploits early)
- [ ] Test pure distance shaping in Stage 0 first (validation)

---

## Git Commit Recommendation

```powershell
git add .
git commit -m "Stage A: Document 5 failed attempts - heading reward exploit

Attempts:
1. SAC: 0% (wrong algorithm)
2. PPO high entropy: 40%→0% (heading exploit + variance)
3. PPO low entropy: 40%→0% (variance explosion)
4. PPO high penalties: 30-60% stable (BEST but still exploits)
5. PPO conditional: 18%→5% (reward mismatch killed it)

Root cause: Heading reward exploitable regardless of conditioning.
Solution: Remove heading entirely, use pure distance shaping.
Next: Train from scratch with simplified reward structure."
```

---
