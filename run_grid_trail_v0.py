import gymnasium as gym
from gymnasium_env.grid_trail import GridTrailRenderEnv

gym.register(
    id="gymnasium_env/GridTrail-v0",
    entry_point=GridTrailRenderEnv,
)

env = gym.make("gymnasium_env/GridTrail-v0", render_mode="human", size=40)

(state, _) = env.reset()
done = False
steps = 0

MAX_TIMESTEPS = 500

while not done and steps < MAX_TIMESTEPS:
    action = env.action_space.sample()
    (next_state, reward, terminated, truncated, info) = env.step(action)
    print(f"Step: {steps}, Action: {action}, Reward: {reward}, Next State: {next_state}, Info: {info}")
    done = terminated or truncated
    steps += 1

env.close()
