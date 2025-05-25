import numpy as np
from collections import deque
from env.grid_trail import GridTrailParallelEnv
from DeepQLearning import DeepQLearning, Trainer, build_model, build_agents
import os

# --- Parameters ---
size = 40
num_agents = 5 
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 11
batch_size = 64
memory_size = 20000
max_steps = 100

net_params = {
    'gamma': gamma,
    'epsilon': epsilon,
    'epsilon_min': epsilon_min,
    'epsilon_decay': epsilon_decay,
    'batch_size': batch_size,
    'memory_size': memory_size,
} 

# --- Env Parameters ---

env_params = {
    'render_mode': None,
    'size': size,
    'num_agents': num_agents,
    'flatten_observations': True,
    'reward': None
}

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Train all agents ---
for reward_function in ['v0', 'v1', 'v2']:
    os.makedirs(f'results/{reward_function}/', exist_ok=True)
    os.makedirs(f'models/{reward_function}/', exist_ok=True)

    env_params['reward'] = reward_function
    env = GridTrailParallelEnv(**env_params)
    env.reset()
    learners = build_agents(env=env, **net_params)

    trainer = Trainer(env=env, learners=learners, max_steps=max_steps)
    coverage = []
    strawberry = []
    for episode in range(episodes):
        print(f"\n--- Episode {episode+1}/{episodes} ---")
        trainer.train()
        observations,cov_pct,found = env.reset()
        coverage.append(cov_pct)
        strawberry.append(found)
        if episode % 10 == 0:
            env.write_rewards(f'results/{reward_function}/rewards_{reward_function}.csv')
            env.write_coverage(f'results/{reward_function}/coverage_{reward_function}.csv',coverage_list=coverage)
            trainer.save_models(f'models/{reward_function}/')