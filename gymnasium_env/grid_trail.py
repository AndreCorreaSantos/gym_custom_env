from typing import Optional
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
import pygame

class GridTrailRenderEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 10, num_agents: int = 4, trail_lifetime: int = 50):
        self.size = size
        self.window_size = 1000
        self.num_agents = num_agents
        self.trail_lifetime = trail_lifetime  # Number of steps before trail disappears

        # Store multiple agents' locations
        self._agent_locations = [np.array([-1, -1], dtype=np.int32) for _ in range(num_agents)]
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observation space: Tuple of Box spaces for each agent
        single_obs_space = spaces.Box(
            low=0, high=4, shape=(5, 5), dtype=np.int32
        )
        self.observation_space = spaces.Tuple([single_obs_space for _ in range(num_agents)])

        # Action space: one Discrete(4) per agent
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(4) for _ in range(num_agents)])
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        self._trail = []  # List of (position, lifetime) tuples

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

       
    def _get_obs(self, agent_idx: int):
        obs = np.zeros((5, 5), dtype=np.int32)

        agent_x, agent_y = self._agent_locations[agent_idx]
        target_x, target_y = self._target_location

        # Populate the 5x5 grid
        for i in range(5):
            for j in range(5):
                # map local 5x5 coordinates to global grid coordinates
                grid_x = agent_x + (i - 2) 
                grid_y = agent_y + (j - 2)

                # Check if within grid bounds
                if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
                    # Priority: target > observing agent > other agent > trail > empty
                    if grid_x == target_x and grid_y == target_y:
                        obs[i, j] = 4  # Target
                    elif grid_x == agent_x and grid_y == agent_y:
                        obs[i, j] = 3  # Observing agent
                    elif any(
                        np.array_equal([grid_x, grid_y], loc)
                        for idx, loc in enumerate(self._agent_locations) if idx != agent_idx
                    ):
                        obs[i, j] = 2  # Other agent
                    elif any(
                        np.array_equal([grid_x, grid_y], pos) and lifetime > 0
                        for pos, lifetime in self._trail
                    ):
                        obs[i, j] = 1  # Trail

        return obs

    def _get_info(self):
        # Get observations for each agent
        observations = [self._get_obs(i) for i in range(self.num_agents)]

        return {
            "distances": [
                np.linalg.norm(loc - self._target_location, ord=1)
                for loc in self._agent_locations
            ],
            "size": self.size,
            "observations": observations  # 5x5 observation grid for each agent
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # randomly assign agent locations
        self._agent_locations = [
            self.np_random.integers(0, self.size, size=2, dtype=int)
            for _ in range(self.num_agents)
        ]
        # ensure agents dont overlap
        for i in range(self.num_agents):
            while any(
                np.array_equal(self._agent_locations[i], self._agent_locations[j])
                for j in range(self.num_agents) if i != j
            ):
                self._agent_locations[i] = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._trail = []

        # initialize target
        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while any(np.array_equal(self._target_location, loc) for loc in self._agent_locations):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # observations for all agents
        observations = tuple(self._get_obs(i) for i in range(self.num_agents))
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, info

    def step(self, actions):

        assert len(actions) == self.num_agents, "Must provide actions for all agents"

        # update each agent position
        for i, action in enumerate(actions):
            direction = self._action_to_direction[int(action)]
            self._agent_locations[i] = np.clip(
                self._agent_locations[i] + direction, 0, self.size - 1
            )

        # decay trails
        i = 0
        while i < len(self._trail):
            pos, lifetime = self._trail[i]
            lifetime -= 1
            self._trail[i] = (pos, lifetime)
            if lifetime <= 0:
                self._trail.pop(i)
            else:
                i += 1

        # add new trail positions
        for loc in self._agent_locations:
            self._trail.append((loc.copy(), self.trail_lifetime))

        # Check termination
        terminated = any(
            np.array_equal(loc, self._target_location) for loc in self._agent_locations
        )
        truncated = False
        # Compute shared reward (1.0 if any agent reaches target, else 0.0)
        reward = 1.0 if terminated else 0.0
        observations = tuple(self._get_obs(i) for i in range(self.num_agents))
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

            # Load the image once
            self.cell_size = self.window_size // self.size
            self.agent_image = pygame.image.load("gymnasium_env/ant.jpg").convert_alpha()
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))

            self.target_img = pygame.image.load("gymnasium_env/strawberry.png")
            self.target_img = pygame.transform.scale(self.target_img, (self.cell_size,self.cell_size))


        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background


        # Draw grid
        for x in range(self.size):
            for y in range(self.size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(canvas, (200, 200, 200), rect, width=1)

        # Draw trail (faded color)
        for (x, y), lifetime in self._trail:
            if lifetime > 0:
                color = (100, 255, 100, max(50, int(255 * lifetime / self.trail_lifetime)))
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                s.fill(color)
                canvas.blit(s, (x * self.cell_size, y * self.cell_size))

        # Draw target (red square)
        tx, ty = self._target_location
        canvas.blit(self.target_img, (tx*self.cell_size, ty*self.cell_size))

        # Draw agents using image
        for loc in self._agent_locations:
            x, y = loc
            canvas.blit(self.agent_image, (x * self.cell_size, y * self.cell_size))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()