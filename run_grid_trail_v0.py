from pettingzoo.utils import parallel_to_aec
from gymnasium_env.grid_trail import GridTrailParallelEnv 
import numpy as np

env = GridTrailParallelEnv(render_mode="human", size=40,num_agents=50)
env.reset()



done = {agent: False for agent in env.agents}
steps = 0
MAX_TIMESTEPS = 1000

while not any(done.values()) and steps < MAX_TIMESTEPS:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
        if not done[agent]
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Step: {steps}")
    print(observations)
    env.render()
    # print(f"\nStep: {steps}")
    # for agent in env.agents:
        # print(f"{agent}: Action: {actions.get(agent)}, Reward: {rewards[agent]}, Done: {terminations[agent] or truncations[agent]}")

    done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
    steps += 1

env.close()
