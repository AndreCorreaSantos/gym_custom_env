from typing import Optional
import numpy as np
import pygame
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec
from gymnasium import spaces

class GridTrailParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_agents=4, trail_lifetime=50):
        self.size = size
        self.window_size = 1000
        self._num_agents = num_agents
        self.trail_lifetime = trail_lifetime
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]

        self._agent_locations = [np.array([-1, -1], dtype=np.int32) for _ in range(num_agents)]
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._trail = []

        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

        single_obs_space = spaces.Box(low=0, high=4, shape=(5, 5), dtype=np.int32)
        single_action_space = spaces.Discrete(4)

        self.observation_spaces = {agent: single_obs_space for agent in self.agents}
        self.action_spaces = {agent: single_action_space for agent in self.agents}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self, agent_idx):
        obs = np.zeros((5, 5), dtype=np.int32)
        agent_x, agent_y = self._agent_locations[agent_idx]
        target_x, target_y = self._target_location

        for i in range(5):
            for j in range(5):
                grid_x = agent_x + (i - 2)
                grid_y = agent_y + (j - 2)

                if 0 <= grid_x < self.size and 0 <= grid_y < self.size:
                    if grid_x == target_x and grid_y == target_y:
                        obs[i, j] = 4
                    elif grid_x == agent_x and grid_y == agent_y:
                        obs[i, j] = 3
                    elif any(np.array_equal([grid_x, grid_y], loc) for idx, loc in enumerate(self._agent_locations) if idx != agent_idx):
                        obs[i, j] = 2
                    elif any(np.array_equal([grid_x, grid_y], pos) and lifetime > 0 for pos, lifetime in self._trail):
                        obs[i, j] = 1
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.agents = self.possible_agents[:]
        self.np_random = np.random.default_rng(seed)

        self._agent_locations = [
            self.np_random.integers(0, self.size, size=2, dtype=int)
            for _ in range(self.num_agents)
        ]

        for i in range(self.num_agents):
            while any(np.array_equal(self._agent_locations[i], self._agent_locations[j]) for j in range(self.num_agents) if i != j):
                self._agent_locations[i] = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        while any(np.array_equal(self._target_location, loc) for loc in self._agent_locations):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._trail = []

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        return observations

    def step(self, actions):
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Update positions and record trails
        new_locations = self._agent_locations[:]
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            move = self._action_to_direction[action]
            new_loc = self._agent_locations[i] + move
            new_loc = np.clip(new_loc, 0, self.size - 1)
            self._trail.append((self._agent_locations[i].copy(), self.trail_lifetime))
            new_locations[i] = new_loc

        self._agent_locations = new_locations

        # Update trail lifetimes
        self._trail = [(pos, lifetime - 1) for pos, lifetime in self._trail if lifetime > 1]

        # Rewards: +1 if agent is on target
        for i, agent in enumerate(self.agents):
            if np.array_equal(self._agent_locations[i], self._target_location):
                rewards[agent] = 1.0
                terminations[agent] = True

        observations = {agent: self._get_obs(i) for i, agent in enumerate(self.agents)}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "human":
            return

        cell_size = self.window_size // self.size  # Size of each grid cell

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))  # Clear screen

        # Draw trail
        for (pos, lifetime) in self._trail:
            if lifetime > 0:
                rect = pygame.Rect(pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.window, (100, 100, 100), rect)

        # Draw target
        rect = pygame.Rect(
            self._target_location[0] * cell_size,
            self._target_location[1] * cell_size,
            cell_size,
            cell_size
        )
        pygame.draw.rect(self.window, (255, 0, 0), rect)

        # Draw agents
        for loc in self._agent_locations:
            rect = pygame.Rect(loc[0] * cell_size, loc[1] * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.window, (0, 255, 0), rect)

        # Optional: draw grid lines
        for x in range(self.size):
            pygame.draw.line(self.window, (50, 50, 50), (x * cell_size, 0), (x * cell_size, self.window_size))
        for y in range(self.size):
            pygame.draw.line(self.window, (50, 50, 50), (0, y * cell_size), (self.window_size, y * cell_size))

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
