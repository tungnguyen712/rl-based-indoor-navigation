import time
import gymnasium as gym
import pybullet as p
import numpy as np
import pybullet_data
from typing import Any

import json
import os
class IndoorMazeEnv(gym.Env):
    metadata = {
        "render_modes": ["direct", "gui"],
        "render_fps": 60,
    }

    def __init__(self, maze_layouts, render_mode="direct",
                 terminate_on_collision=True,
                 max_collisions_per_episode=3,
                 debug=False):
        super().__init__()
        self.maze_layouts = maze_layouts
        self.grid: Any = None
        self.wall_ids = []
        self.physics_steps_per_action = 12
        
        self.wall_height = 0.4
        self.lidar_num_rays = 64
        self.lidar_fov = 180
        self.lidar_max_range = 3.0
        self.lidar_debug_line_ids = []
        self.debug_lidar_every = 10

        self.max_episode_steps = 2500
        self.goal_radius = 0.75
        self.min_spawn_dist = 1.5
        self.goal_marker_radius = 0.6
        
        # Track previous action for observation
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        # collision behavior flags
        self.terminate_on_collision = terminate_on_collision
        self.max_collisions_per_episode = max_collisions_per_episode
        self.debug = debug

        # differential drive with [left_wheel_speed, right_wheel_speed]
        # both wheels can go forward/backward independently -> can turn in place
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        # obs space = [LIDAR(64) + goal_dist(1) + goal_angle(1) + velocity(3) + prev_action(2)] = 71 dims
        obs_dim = self.lidar_num_rays + 2 + 3 + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # rewards
        self.goal_reward = 10.0
        self.dist_weight = 1.0
        self.heading_weight = 0.005
        self.collision_penalty = -5.0
        self.time_penalty = -0.01
        self.backward_penalty = -0.5

        # differential drive parameters
        self.max_wheel_speed = 10.0  # rad/sec
        self.wheel_radius = 0.05  # meters
        self.wheel_base = 0.3  # distance between wheels (meters)

        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        if p.isConnected():
            p.disconnect()

        if self.render_mode == "gui":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # basic pybullet setup
        p.setRealTimeSimulation(0)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # load robot
        self.robot_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.1], globalScaling=0.5)

        # create goal marker (VISUAL ONLY - no collision)
        vis_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.goal_marker_radius,
            rgbaColor=[0.0, 1.0, 0.0, 0.2],
        )

        self.goal_marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # no collision shape
            baseVisualShapeIndex=vis_shape,
            basePosition=[0, 0, self.goal_marker_radius],
        )
        
        # add strong green outline for visibility
        p.changeVisualShape(self.goal_marker_id, -1, rgbaColor=[0.0, 1.0, 0.0, 0.2])
        p.changeVisualShape(self.goal_marker_id, -1, specularColor=[0.0, 1.0, 0.0])

        # simplify GUI and set up camera
        if self.render_mode == "gui":
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            # set initial camera position on birdseye view
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[2.0, 2.0, 0]
            )

        # get joint ids
        self.rear_wheel_joints = []
        self.front_wheel_joints = []
        self.steering_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            if joint_name in ["left_rear_wheel_joint", "right_rear_wheel_joint"]:
                self.rear_wheel_joints.append(i)
            elif joint_name in ["left_front_wheel_joint", "right_front_wheel_joint"]:
                self.front_wheel_joints.append(i)
            elif joint_name in ["left_steering_hinge_joint", "right_steering_hinge_joint"]:
                self.steering_joints.append(i)
        

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.collision_count = 0
        self.collided_this_step = False
        self.succeeded_this_step = False
        self.prev_collided = False
        self.prev_goal_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)  # reset previous action

        # pick a maze layout
        maze: Any = self.np_random.choice(self.maze_layouts)
        self.grid = maze["grid"]
        self.cell_size = maze.get("cell_size", 0.5)

        # reset walls
        self.remove_walls()
        self.create_walls()

        max_spawn_tries = 20
        spawn_successful = False
        
        for attempt in range(max_spawn_tries):
            # get start + goal positions
            self.start_pos, self.goal_pos = self.sample_random_position()
            
            # enforce minimum distance between spawn and goal
            if np.linalg.norm(np.array(self.start_pos) - np.array(self.goal_pos)) < self.min_spawn_dist:
                continue

            yaw_options = [0, np.pi/2, np.pi, -np.pi/2]
            spawn_yaw = self.np_random.choice(yaw_options)
            
            # reset robot position, orientation, and velocity
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.start_pos[0], self.start_pos[1], 0.1],
                p.getQuaternionFromEuler([0, 0, spawn_yaw]),
            )
            p.resetBaseVelocity(self.robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

            # stop all motors
            for i in self.rear_wheel_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0
                )

            for i in self.steering_joints:
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=0,
                    force=0
                )
            
            # step simulation to settle physics
            for _ in range(10):
                p.stepSimulation()
            
            if not self.check_collision():
                spawn_successful = True
                break
        
        if not spawn_successful and self.debug:
            print(f"[WARNING] Failed to find collision-free spawn after {max_spawn_tries} attempts")

        # reset goal marker position
        p.resetBasePositionAndOrientation(
            self.goal_marker_id,
            [self.goal_pos[0], self.goal_pos[1], self.goal_marker_radius],
            [0, 0, 0, 1],
        )

        # reset lidar debug lines
        if self.render_mode == "gui":
            for line_id in self.lidar_debug_line_ids:
                p.removeUserDebugItem(line_id)
            self.lidar_debug_line_ids = []
            
            # Update camera to focus on the current maze
            maze_center_x = (len(self.grid[0]) * self.cell_size) / 2
            maze_center_y = (len(self.grid) * self.cell_size) / 2
            p.resetDebugVisualizerCamera(
                cameraDistance=max(len(self.grid), len(self.grid[0])) * self.cell_size * 0.8,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[maze_center_x, maze_center_y, 0]
            )

        obs = self.get_observation()
        
        # initialize prev_goal_dist to improve early shaping stability
        self.prev_goal_dist = np.linalg.norm(np.array(self.start_pos) - np.array(self.goal_pos))

        info = {}

        return obs, info

    def step(self, action):
        # Store action for next observation
        self.prev_action = np.array(action, dtype=np.float32)
        
        self.control_robot(action)
        for _ in range(self.physics_steps_per_action):
            p.stepSimulation()

        current_collision = self.check_collision()
        self.collided_this_step = current_collision and not self.prev_collided
        
        if self.collided_this_step:
            self.collision_count += 1
        
        self.prev_collided = current_collision
        
        obs = self.get_observation()
        
        success = self.check_success()
        self.succeeded_this_step = success
        
        reward = self.compute_reward()
        
        if self.terminate_on_collision:
            # if collided, terminate immediately
            terminated = success or self.collided_this_step
        else:
            terminated = success or (self.collision_count >= self.max_collisions_per_episode)
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        
        info = {"is_success": success}
        
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected():
            p.disconnect()

    def create_walls(self):
        grid_rows = len(self.grid)
        grid_cols = len(self.grid[0])
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if self.grid[row][col] == 1:
                    # distinguish boundary walls from inner obstacles
                    is_boundary = (row == 0 or row == grid_rows - 1 or 
                                 col == 0 or col == grid_cols - 1)
                    
                    if is_boundary:
                        # boundary walls: full cell size (1.0×) with no gaps
                        wall_width = self.cell_size * 1.0
                        wall_depth = self.cell_size * 1.0
                        color = [0.7, 0.7, 0.7, 1]  # Gray
                    else:
                        # inner obstacles (0.35× cell_size) matches with research paper design and gives robot room to turn
                        wall_width = self.cell_size * 0.35
                        wall_depth = self.cell_size * 0.35
                        color = [0.5, 0.5, 0.5, 1]
                    
                    # create wall collision and visual shapes
                    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2])
                    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2], rgbaColor=color)

                    # create wall body
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape_id,
                        baseVisualShapeIndex=visual_shape_id,
                        basePosition=[
                            col * self.cell_size + self.cell_size / 2,
                            row * self.cell_size + self.cell_size / 2,
                            self.wall_height / 2,
                        ]
                    )
                    self.wall_ids.append(body_id)

    def remove_walls(self):
        for wall_id in self.wall_ids:
            p.removeBody(wall_id)
        self.wall_ids = []

    def sample_random_position(self, min_dist=1, max_tries=100):
        # Find free cells with sufficient wall clearance
        min_wall_clearance = 0.4  # min 0.4m turning radius for robots to turn around
        free_cells_with_clearance = []
        
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 0:  # Free cell
                    # calculate cell center position
                    pos_x = col * self.cell_size + self.cell_size / 2
                    pos_y = row * self.cell_size + self.cell_size / 2
                    
                    # check clearance to nearest wall
                    has_clearance = True
                    for wall_row in range(len(self.grid)):
                        for wall_col in range(len(self.grid[0])):
                            if self.grid[wall_row][wall_col] == 1:  # Wall cell
                                wall_x = wall_col * self.cell_size + self.cell_size / 2
                                wall_y = wall_row * self.cell_size + self.cell_size / 2
                                dist_to_wall = np.sqrt((pos_x - wall_x)**2 + (pos_y - wall_y)**2)
                                
                                if dist_to_wall < min_wall_clearance:
                                    has_clearance = False
                                    break
                        if not has_clearance:
                            break
                    
                    if has_clearance:
                        free_cells_with_clearance.append((row, col))
        
        # fall back to all free cells if clearance filtering is too strict
        if len(free_cells_with_clearance) < 2:
            free_cells_with_clearance = [(row, col) for row in range(len(self.grid)) 
                                         for col in range(len(self.grid[0])) 
                                         if self.grid[row][col] == 0]
        
        start_cell = self.np_random.choice(free_cells_with_clearance)
        start_pos = (
            start_cell[1] * self.cell_size + self.cell_size / 2,
            start_cell[0] * self.cell_size + self.cell_size / 2,
        )

        for _ in range(max_tries):
            goal_cell = self.np_random.choice(free_cells_with_clearance)
            goal_pos = (
                goal_cell[1] * self.cell_size + self.cell_size / 2,
                goal_cell[0] * self.cell_size + self.cell_size / 2,
            )
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
            if dist >= min_dist:
                return start_pos, goal_pos
        
        # fallback: accept any position with sufficient distance
        while True:
            goal_cell = self.np_random.choice(free_cells_with_clearance)
            if tuple(goal_cell) != tuple(start_cell):
                goal_pos = (
                    goal_cell[1] * self.cell_size + self.cell_size / 2,
                    goal_cell[0] * self.cell_size + self.cell_size / 2,
                )
                return start_pos, goal_pos

    def control_robot(self, action):
        left_speed, right_speed = action
        
        # convert normalized action [-1, 1] to wheel velocities
        left_vel = left_speed * self.max_wheel_speed
        right_vel = right_speed * self.max_wheel_speed
        
        # apply to rear wheels as differential drive
        # left wheel (joint 2)
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.rear_wheel_joints[0],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_vel,
            force=200,  # increased force for better control
        )
        # right wheel (joint 3)
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.rear_wheel_joints[1],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_vel,
            force=200,
        )
        
        # lock steering joints at 0 (straight) for differential drive
        for joint_index in self.steering_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force=50,
            )

    def get_observation(self):
        # get robot position, yaw, and velocities
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y, robot_z = position
        euler = p.getEulerFromQuaternion(orientation)
        yaw = euler[2]
        
        # Get velocities
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        vx, vy = linear_vel[0], linear_vel[1]
        wz = angular_vel[2]
        
        # Normalize velocities (assume max ~2 m/s linear, ~2 rad/s angular)
        norm_vx = np.clip(vx / 2.0, -1.0, 1.0)
        norm_vy = np.clip(vy / 2.0, -1.0, 1.0)
        norm_wz = np.clip(wz / 2.0, -1.0, 1.0)

        # generate lidar ray angles centered on robot yaw spanning lidar_fov
        angles = np.linspace(-self.lidar_fov/2, self.lidar_fov/2, self.lidar_num_rays) *np.pi/180
        ray_from = []
        ray_to = []
        for a in angles:
            theta = yaw + a
            dx = np.cos(theta)
            dy = np.sin(theta)

            ray_from.append([robot_x, robot_y, robot_z + 0.1])
            ray_to.append([
                robot_x + dx * self.lidar_max_range,
                robot_y + dy * self.lidar_max_range,
                robot_z + 0.1
            ])

        # raycast each direction up to lidar_max_range, normalize distances to [0, 1]
        results = p.rayTestBatch(ray_from, ray_to)
        # reset lidar debug lines
        # if self.render_mode == "gui" and self.step_count % self.debug_lidar_every == 0:
        #     # remove previous lines
        #     for line_id in self.lidar_debug_line_ids:
        #         p.removeUserDebugItem(line_id)
        #     self.lidar_debug_line_ids = []
        lidar_distances = []

        for i, r in enumerate(results):
            hit_body = r[0]
            hit_fraction = r[2]

            # ignore self robot hits
            if hit_body == self.robot_id:
                hit_fraction = 1.0

            lidar_distances.append(hit_fraction)

            # show lidar debug lines
            # if self.render_mode == "gui" and self.step_count % self.debug_lidar_every == 0:
            #     # compute endpoint
            #     fx, fy, fz = ray_from[i]
            #     tx, ty, tz = ray_to[i]

            #     end = [
            #         fx + (tx - fx) * hit_fraction,
            #         fy + (ty - fy) * hit_fraction,
            #         fz + (tz - fz) * hit_fraction,
            #     ]
            #     line_id = p.addUserDebugLine(
            #         ray_from[i],
            #         end,
            #         lineColorRGB=[1, 0, 0],  # red
            #         lineWidth=1,
            #         lifeTime=0,  # persists until removed
            #     )
            #     self.lidar_debug_line_ids.append(line_id)
        norm_lidar = np.array(lidar_distances, dtype=np.float32)

        # compute goal distance and normalize to [0, 1]
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))
        max_dist = np.linalg.norm(np.array([len(self.grid)*self.cell_size, len(self.grid[0])*self.cell_size]))
        norm_goal_dist = goal_dist / max_dist
        norm_goal_dist = np.clip(norm_goal_dist, 0.0, 1.0)

        # compute goal angle relative to robot's heading, wrap to [-pi, pi], normalize to [-1, 1]
        goal_angle = np.arctan2(goal_dy, goal_dx)
        relative_angle = goal_angle - yaw
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
        norm_goal_angle = relative_angle / np.pi

        # concatenate: lidar(64) + goal_dist(1) + goal_angle(1) + velocity(3) + prev_action(2) = 71 dims
        observation = np.concatenate([
            norm_lidar,
            np.array([norm_goal_dist, norm_goal_angle], dtype=np.float32),
            np.array([norm_vx, norm_vy, norm_wz], dtype=np.float32),
            self.prev_action
        ])
        return observation


    def compute_reward(self):
        # get robot position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        euler = p.getEulerFromQuaternion(orientation)
        robot_yaw = euler[2]

        # get goal distance and angle
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))
        goal_angle = np.arctan2(goal_dy, goal_dx)

        reward = 0.0
        
        if self.prev_goal_dist is not None:
            linear_vel, _ = p.getBaseVelocity(self.robot_id)
            robot_forward = [np.cos(robot_yaw), np.sin(robot_yaw)]
            forward_speed = linear_vel[0] * robot_forward[0] + linear_vel[1] * robot_forward[1]
            
            # distance shaping reward
            dist_delta = self.prev_goal_dist - goal_dist
            
            if dist_delta >= 0:
                # moving closer to goal
                if forward_speed >= -0.05:
                    # moving forward toward goal: FULL reward
                    reward = self.dist_weight * dist_delta
                else:
                    # driving backwards toward goal: NO reward (blind spot)
                    reward = 0.0
            else:
                # moving away from goal
                # gentle penalty to allows turns without harsh punishment
                reward = self.dist_weight * dist_delta * 0.2
            
            # penalty for driving backwards (blind spot of our 180° front lidar)
            if forward_speed < -0.1:
                reward += self.backward_penalty  # -0.5 per step
        
        self.prev_goal_dist = goal_dist
        
        # time penalty
        reward += self.time_penalty  # -0.005 per step
        
        if getattr(self, "succeeded_this_step", False):
            # success bonus
            reward += self.goal_reward
        
        # collision penalty
        if getattr(self, "collided_this_step", False):
            reward += self.collision_penalty  # -5.0
        
        return reward

    def check_success(self):
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        dist = np.linalg.norm(np.array([robot_x, robot_y]) - np.array(self.goal_pos))
        return dist <= self.goal_radius

    def check_collision(self):
        contacts = p.getContactPoints(bodyA=self.robot_id)
        wall_ids_set = set(self.wall_ids)
        for contact in contacts:
            bodyA_id = contact[1]
            bodyB_id = contact[2]
            if bodyA_id == self.robot_id:
                if bodyB_id in wall_ids_set:
                    return True
            else:
                if bodyA_id in wall_ids_set:
                    return True
        return False
    

def main():
    # load mazes from json files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    maze_dir = os.path.join(project_root, "assets", "train")

    # choose a small subset first for debugging
    maze_files = [
        "train_maze_01.json",
    ]

    maze_layouts = []
    for f in maze_files:
        path = os.path.join(maze_dir, f)
        with open(path, "r") as fp:
            maze_layouts.append(json.load(fp))

    # create env in GUI mode for visual debugging
    env = IndoorMazeEnv(maze_layouts=maze_layouts, render_mode="gui")

    try:
        # repeated reset test
        for ep in range(10):
            obs, info = env.reset(seed=ep)

            print(f"\nEpisode {ep} reset")
            print("obs shape:", obs.shape, "dtype:", obs.dtype)
            print("num walls:", len(env.wall_ids))
            print("start:", env.start_pos, "goal:", env.goal_pos)

            # random actions
            for t in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if t % 20 == 0:
                    pos, _ = p.getBasePositionAndOrientation(env.robot_id)
                    print("Reward:", reward)
                    print("Terminated:", terminated, "\tTruncated:", truncated)
                    print("Robot pos: ", pos[0], pos[1])
                    print("Goal dist (normed)", obs[env.lidar_num_rays])

                if terminated or truncated:
                    print(f"Episode ended at t={t} (terminated={terminated}, truncated={truncated})")
                    break
                time.sleep(1.0 / 60.0)
    finally:
        env.close()

if __name__ == "__main__":
    main()