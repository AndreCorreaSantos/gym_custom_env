
#
# python train_grid_world_render_v0.py <train|test>
#

import gymnasium as gym
from gymnasium_env.grid_world_render import GridWorldRenderEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import sys
# train = True if sys.argv[1] == 'train' else False
train = True

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldRenderEnv,
)

if train:
    env = gym.make("gymnasium_env/GridWorld-v0", size=5, render_mode="rgb_array")
    env = FlattenObservation(env)
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    new_logger = configure('log/ppo_custom_env', ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=100_000)
    model.save("data/ppo_custom_env")
    print('model trained')

print('loading model')
model = PPO.load("data/ppo_custom_env")
env = gym.make("gymnasium_env/GridWorld-v0", size=5, render_mode="human")
env = FlattenObservation(env)
(obs, _) = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action.item())
    print(f"Action: {action}, Reward: {reward}, Next State: {obs}")