import gymnasium as gym
from gymnasium_env.grid_trail import GridTrailRenderEnv
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from gymnasium import Env
from gymnasium.spaces import Box
import torch

# --- DummyEnv to satisfy SB3 API ---
class DummyEnv(Env):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

# --- Register and create your environment ---
gym.register(
    id="gymnasium_env/GridTrail-v0",
    entry_point=GridTrailRenderEnv,
)

num_agents = 4
env = gym.make("gymnasium_env/GridTrail-v0", render_mode=None, size=40, num_agents=num_agents)

# --- Create models and buffers ---
models = []
buffers = []

for i in range(num_agents):
    obs_space = env.observation_space.spaces[i]
    act_space = env.action_space.spaces[i]
    
    # Verify observation space is Box with shape (5,5)
    if not isinstance(obs_space, Box) or obs_space.shape != (5, 5):
        print(f"Warning: Agent {i} observation_space is {obs_space}. Expected Box(shape=(5,5)). Fixing...")
        obs_space = Box(low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32)

    dummy_env = DummyEnv(obs_space, act_space)

    model = DQN(
        policy="MlpPolicy",
        env=dummy_env,
        learning_starts=1000,
        buffer_size=50000,
        batch_size=32,
        train_freq=1,
        verbose=0,
        exploration_fraction=0.5,  # Explore for 50% of training
        exploration_final_eps=0.1,  # End with 10% random actions
        learning_rate=1e-4,  # Lower learning rate for stability
    )
    # Set up logger for the model
    # model.logger = configure(f"logs/agent_{i}/", ["stdout", "csv"])  # Log to console and CSV
    models.append(model)

    buffer = ReplayBuffer(
        buffer_size=50000,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    buffers.append(buffer)

# --- Training loop ---
total_episodes = 500
max_steps_per_episode = 200

for ep in range(total_episodes):
    obs, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps_per_episode:
        actions = []
        for i in range(num_agents):
            action, _ = models[i].predict(obs[i], deterministic=False)
            actions.append(action)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        shared_reward = np.array([float(rewards)])
        # Extract scalar reward
        total_reward += shared_reward

        # Debug shapes and observations periodically
        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  Rewards: {rewards}, Type: {type(rewards)}, Shape: {np.array(rewards).shape}")
            for i in range(num_agents):
                print(f"  Agent {i} obs: {obs[i].shape}, action: {actions[i]}, next_obs: {next_obs[i].shape}, obs_values: {obs[i].flatten()[:5]}")
                print(f"  Agent {i} add args: obs_shape={obs[i].shape}, action={actions[i]}, next_obs_shape={next_obs[i].shape}, done={done}")

        for i in range(num_agents):
            # Add to custom buffer
            buffers[i].add(
                obs=obs[i],
                action=actions[i],
                reward=shared_reward,
                next_obs=next_obs[i],
                done=done,
                infos=None
            )
            # Add to model's internal replay buffer
            models[i].replay_buffer.add(
                obs=obs[i],
                action=actions[i],
                reward=shared_reward,
                next_obs=next_obs[i],
                done=done,
                infos={}
            )

            if buffers[i].size() > models[i].learning_starts:
                models[i].train(gradient_steps=1, batch_size=models[i].batch_size)  # Train without external batch

        obs = next_obs
        step += 1

    print(f"[Episode {ep+1}] Steps: {step}, Total Reward: {total_reward}")

# --- Save trained models ---
for i in range(num_agents):
    models[i].save(f"dqn_agent_{i}.zip")

env.close()
print("Training complete.")