import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium_env.grid_trail import GridTrailRenderEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
import numpy as np
import sys
import os

# Custom wrapper to handle multiple PPO models with shared rewards
class MultiAgentPPOWrapper(gym.Wrapper):
    def __init__(self, env, num_agents, size, trail_lifetime):
        super().__init__(env)
        self.num_agents = num_agents
        self.size = size
        self.trail_lifetime = trail_lifetime
        
        # Observation space: Concatenate all agents' observations
        single_obs_shape = env.observation_space.spaces[0].shape  # Shape of one agent's observation
        total_obs_size = single_obs_shape[0] * single_obs_shape[1] * num_agents
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(total_obs_size,), dtype=np.int32
        )
        
        # Action space: MultiDiscrete for independent actions
        self.action_space = spaces.MultiDiscrete([4] * num_agents)
        
        # Initialize separate PPO models for each agent
        self.models = []
        for i in range(num_agents):
            # Create a dummy environment for model initialization
            dummy_env = gym.make(
                "gymnasium_env/GridTrailRender-v0",
                size=self.size,
                num_agents=num_agents,
                trail_lifetime=self.trail_lifetime,
                render_mode="rgb_array"
            )
            dummy_env = gym.Wrapper(dummy_env)
            dummy_env.observation_space = env.observation_space.spaces[0]  # Single agent's obs space
            dummy_env.action_space = spaces.Discrete(4)  # Single agent's action space
            model = PPO("MlpPolicy", dummy_env, verbose=0, device="cpu")
            self.models.append(model)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Flatten list of observations
        flat_obs = np.concatenate([o.flatten() for o in obs])
        return flat_obs, info

    def step(self, action):
        # Action is a list of actions [a1, a2, ..., an]
        actions = tuple(action)  # Convert to tuple for env
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        # Flatten observations
        flat_obs = np.concatenate([o.flatten() for o in obs])
        # Shared reward: max of all rewards (1.0 if any agent succeeds, else 0.0)
        shared_reward = max(rewards)
        return flat_obs, shared_reward, terminated, truncated, info

    def learn(self, total_timesteps):
        # Custom training loop
        steps_taken = 0
        while steps_taken < total_timesteps:
            obs, _ = self.env.reset()
            done = False
            while not done and steps_taken < total_timesteps:
                # Collect actions for all agents
                actions = []
                for i, model in enumerate(self.models):
                    single_obs = obs[i * 25:(i + 1) * 25]  # Extract agent's observation
                    action, _ = model.predict(single_obs, deterministic=False)
                    actions.append(action.item())
                
                # Step environment
                next_obs, shared_reward, terminated, truncated, info = self.step(actions)
                steps_taken += 1
                
                # Update each model with the shared reward
                for i, model in enumerate(self.models):
                    single_obs = obs[i * 25:(i + 1) * 25]
                    single_next_obs = next_obs[i * 25:(i + 1) * 25]
                    # Simulate a single-agent experience
                    model.rollout_buffer.reset()
                    model.rollout_buffer.add(
                        obs=single_obs,
                        action=actions[i],
                        reward=shared_reward,
                        episode_start=False,
                        value=0.0,  # Placeholder
                        log_prob=0.0  # Placeholder
                    )
                    model.rollout_buffer.compute_returns_and_advantage(
                        last_values=0.0,  # Placeholder
                        dones=terminated or truncated
                    )
                    model.train()
                
                obs = next_obs
                done = terminated or truncated

    def save(self, path_prefix):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        # Save each model
        for i, model in enumerate(self.models):
            model.save(f"{path_prefix}_agent_{i}")

    def load(self, path_prefix):
        # Load each model with error handling
        for i in range(self.num_agents):
            model_path = f"{path_prefix}_agent_{i}"
            if not os.path.exists(f"{model_path}.zip"):
                raise FileNotFoundError(
                    f"Model file {model_path}.zip not found. Please run training first with 'python train_grid_trail_render.py train'."
                )
            self.models[i] = PPO.load(model_path, device="cpu")

# Check if training or testing mode
train = True if len(sys.argv) > 1 and sys.argv[1] == 'train' else False

# Environment parameters
ENV_SIZE = 10
NUM_AGENTS = 4
TRAIL_LIFETIME = 50

if train:
    # Create environment for training
    env = gym.make(
        "gymnasium_env/GridTrailRender-v0",
        size=ENV_SIZE,
        num_agents=NUM_AGENTS,
        trail_lifetime=TRAIL_LIFETIME,
        render_mode="rgb_array"
    )
    env = MultiAgentPPOWrapper(env, num_agents=NUM_AGENTS, size=ENV_SIZE, trail_lifetime=TRAIL_LIFETIME)
    check_env(env)  # Verify environment compatibility
    # Set up logging
    new_logger = configure('log/ppo_grid_trail', ["stdout", "csv", "tensorboard"])
    # Train all models
    env.learn(total_timesteps=100_000)
    env.save("data/ppo_grid_trail")
    print('Models trained')

# Testing mode
print('Loading models')
env = gym.make(
    "gymnasium_env/GridTrailRender-v0",
    size=ENV_SIZE,
    num_agents=NUM_AGENTS,
    trail_lifetime=TRAIL_LIFETIME,
    render_mode="human"
)
env = MultiAgentPPOWrapper(env, num_agents=NUM_AGENTS, size=ENV_SIZE, trail_lifetime=TRAIL_LIFETIME)
env.load("data/ppo_grid_trail")
obs, _ = env.reset()
done = False

while not done:
    # Predict actions for each agent
    actions = []
    for i, model in enumerate(env.models):
        single_obs = obs[i * 25:(i + 1) * 25]  # Extract agent's observation
        action, _ = model.predict(single_obs, deterministic=True)
        actions.append(action.item())
    obs, reward, done, _, info = env.step(actions)
    print(f"Actions: {actions}, Reward: {reward}, Info: {info}")

env.close()