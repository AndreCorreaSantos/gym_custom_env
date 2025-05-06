import gymnasium as gym
from gymnasium_env.grid_trail import GridTrailRenderEnv
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
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
    dummy_env = DummyEnv(obs_space, act_space)

    model = DQN(
        policy="MlpPolicy",
        env=dummy_env,
        learning_starts=1000,
        buffer_size=50000,
        batch_size=32,
        train_freq=1,
        verbose=0,
    )
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

    while not done and step < max_steps_per_episode:
        actions = []
        for i in range(num_agents):
            action, _ = models[i].predict(obs[i], deterministic=False)
            actions.append(action)

        next_obs, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        # Debug shapes of all inputs
        print(f"Step {step}:")
        print(f"  Rewards: {rewards}, Type: {type(rewards)}, Shape: {np.array(rewards).shape}")
        for i in range(num_agents):
            print(f"  Agent {i} obs: {obs[i].shape}, action: {actions[i]}, next_obs: {next_obs[i].shape}")



        print(f"  Shared reward: {rewards}")
        r_array = np.array([float(rewards)])
        for i in range(num_agents):
            # Debug arguments before add
            print(f"  Agent {i} add args: obs_shape={obs[i].shape}, action={actions[i]},next_obs_shape={next_obs[i].shape}, done={done}")

            # Use keyword arguments for clarity
            buffers[i].add(
                obs=obs[i],
                action=actions[i],
                reward=rewards,
                next_obs=next_obs[i],
                done=done,
                infos=None
            )

            if buffers[i].size() > models[i].learning_starts:
                batch = buffers[i].sample(models[i].batch_size)
                models[i].train_on_batch(batch)

        obs = next_obs
        step += 1

    print(f"[Episode {ep+1}] Steps: {step}")

# --- Save trained models ---
for i in range(num_agents):
    models[i].save(f"dqn_agent_{i}.zip")

env.close()