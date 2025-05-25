from DeepQLearning import DeepQLearning, Evaluator, build_model, build_agents
from collections import deque
import numpy as np
import os
from env.grid_trail import GridTrailParallelEnv

size = 40
num_agents = 5
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 1
batch_size = 64
memory_size = 20000
max_steps = 100

os.makedirs('evaluation_results', exist_ok=True)

for reward_function in ['v0', 'v1', 'v2']:
    env = GridTrailParallelEnv(render_mode="human", size=size, num_agents=num_agents,
                               flatten_observations=True, reward=reward_function)
    env.reset()

    learners = build_agents(env=env, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                             epsilon_decay=epsilon_decay, episodes=episodes, batch_size=batch_size,
                             memory_size=memory_size)

    evaluator = Evaluator(env=env, learners=learners, max_steps=max_steps, max_episodes=episodes)
    evaluator.load_models(f'models/{reward_function}/')
    evaluator.evaluate()
    env.write_rewards(f'evaluation_results/rewards_{reward_function}.csv')
