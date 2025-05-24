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
episodes = 1
batch_size = 64
memory_size = 20000
max_steps = 100


os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Train all agents ---
for reward_function in ['v0', 'v1', 'v2']:

    os.makedirs(f'models/{reward_function}/', exist_ok=True)
    env = GridTrailParallelEnv(render_mode=None, size=size, num_agents=num_agents,flatten_observations=True, reward=reward_function)
    env.reset()
    learners = build_agents(env=env, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min,
                            epsilon_decay=epsilon_decay, episodes=episodes, batch_size=batch_size,
                            memory_size=memory_size)

    trainer = Trainer(env=env, learners=learners, max_steps=max_steps)
    for episode in range(episodes):
        print(f"\n--- Episode {episode+1}/{episodes} ---")
        print(f"reward: {0}")
        trainer.train()
        if episode % 100 == 0:
            env.write_rewards(f'results/rewards_{reward_function}.csv')
    
    trainer.save_models(f'models/{reward_function}/')