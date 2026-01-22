import gymnasium as gym
import pybullet as p
import numpy as np
import random

class IndoorMazeEnv(gym.Env):
    metadata = {
        "render_modes": ["direct", "gui"],
        "render_fps": 60,
    }

    def __init__(self, maze_layouts, robot_id, render_mode="direct"):
        super().__init__()
        self.maze_layouts = maze_layouts
        self.robot_id = robot_id
        self.grid = None
        self.wall_ids = []
        
        self.wall_height = 0.4
        self.lidar_num_rays = 64
        self.lidar_fov = 180
        self.lidar_max_range = 3.0

        self.max_episode_steps = 500
        self.goal_radius = 0.25
        # action = [v, w]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        # [LIDAR distances + goal direction + goal angle]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * self.lidar_num_rays + [0.0, -1.0]),
            high=np.array([1.0] * self.lidar_num_rays + [1.0, 1.0]),
            dtype=np.float32,
        )

        self.max_v = 1.0
        self.max_w = 1.0
        self.goal_reward = 10
        self.collision_penalty = 10
        self.timestep_penalty = 0.01
        self.dist_weight = 1.0

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
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_goal_dist = None

        # pick a maze layout
        maze = self.np_random.choice(self.maze_layouts)
        self.grid = maze["grid"]
        self.cell_size = maze.get("cell_size", 0.5)

        # start + goal positions
        self.start_pos = self.sample_random_position()[0]
        self.goal_pos = self.sample_random_position()[1]

        self.remove_walls()
        self.create_walls()

        obs = self.get_observation()
        info = {}

        return obs, info

    def step(self, action):
        self.control_robot(action)
        obs = self.get_observation()
        reward = self.compute_reward()
        self.step_count += 1
        terminated = self.check_success() or self.check_collision()
        truncated = self.step_count >= self.max_episode_steps
        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected():
            p.disconnect()
    
    def render(self):
        # return current frame to create simulationvideo
        pass

    def create_walls(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[0])):
                if self.grid[row][col] == 1:
                    wall_width = self.cell_size * 0.98
                    wall_depth = self.cell_size * 0.98
                    # Create wall collision and visual shapes
                    collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2])
                    visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_width/2, wall_depth/2, self.wall_height/2], rgbaColor=[0.7, 0.7, 0.7, 1])

                    # Create wall body
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

    def sample_random_position(self):
        free_cells = [(row, col) for row in range(len(self.grid)) for col in range(len(self.grid[0])) if self.grid[row][col] == 0]
        start_cell = self.np_random.choice(free_cells)
        goal_cell = self.np_random.choice(free_cells)
        start_pos = (
            start_cell[1] * self.cell_size + self.cell_size / 2,
            start_cell[0] * self.cell_size + self.cell_size / 2,
        )
        goal_pos = (
            goal_cell[1] * self.cell_size + self.cell_size / 2,
            goal_cell[0] * self.cell_size + self.cell_size / 2,
        )
        dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
        while dist < 1.5:
            goal_cell = self.np_random.choice(free_cells)
            goal_pos = (
                goal_cell[1] * self.cell_size + self.cell_size / 2,
                goal_cell[0] * self.cell_size + self.cell_size / 2,
            )
            dist = np.linalg.norm(np.array(start_pos) - np.array(goal_pos))
        return start_pos, goal_pos

    def control_robot(self, action):
        pass

    def get_observation(self):
        pass
        # get robot position and yaw (heading)
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y, robot_z = position
        euler = p.getEulerFromQuaternion(orientation)
        yaw = euler[2]

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
        lidar_distances = []

        for r in results:
            hit_fraction = r[2]
            dist = hit_fraction * self.lidar_max_range
            lidar_distances.append(dist / self.lidar_max_range)
        norm_lidar = np.array(lidar_distances, dtype=np.float32)

        # compute goal distance and normalize to [0, 1]
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))
        max_dist = np.linalg.norm(np.array([len(self.grid)*self.cell_size, len(self.grid[0])*self.cell_size]))
        norm_goal_dist = goal_dist / max_dist

        # compute goal angle relative to robot's heading, wrap to [-pi, pi], normalize to [-1, 1]
        goal_angle = np.arctan2(goal_dy, goal_dx)
        relative_angle = goal_angle - yaw
        while relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        while relative_angle < -np.pi:
            relative_angle += 2 * np.pi
        norm_goal_angle = relative_angle / np.pi

        return (norm_lidar, norm_goal_dist, norm_goal_angle)

    def compute_reward(self):
        # get robot position
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]

        # get goal distance
        goal_dx = self.goal_pos[0] - robot_x
        goal_dy = self.goal_pos[1] - robot_y
        goal_dist = np.linalg.norm(np.array([goal_dx, goal_dy]))

        reward = 0.0
        # progress based reward
        if self.prev_goal_dist is not None:
            dist_delta = self.prev_goal_dist - goal_dist
        else:
            dist_delta = 0.0
        reward += self.dist_weight * dist_delta
        self.prev_goal_dist = goal_dist

        # terminal rewards/penalties
        if self.check_success():
            reward += self.goal_reward
        if self.check_collision():
            reward -= self.collision_penalty
        
        # timestep penalty
        reward -= self.timestep_penalty
        return reward

    def check_success(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
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



