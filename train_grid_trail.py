from env.grid_trail import GridTrailParallelEnv
from DeepQLearning import Trainer, build_agents
import os

# --- Parameters ---
size = 40
num_agents = 5 
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
episodes = 10



batch_size = 64
memory_size = 20000
max_steps = 100

target_episode = int(episodes * 0.8)  # 80% point
total_steps = target_episode * max_steps
epsilon_decay = (epsilon_min / epsilon) ** (1 / total_steps)

print(f'Epsilon Decay: {epsilon_decay:.6f}')
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
    reward_dict = {}
    
    for agent in env.agents:
        reward_dict[agent] = []

    for episode in range(episodes):

        print(f'Epsilon: {trainer.learners["agent_0"].epsilon:.4f}')
        ep_rewards,found,cov_pct = trainer.train()

        for agent in env.agents:
            reward_dict[agent].append(ep_rewards[agent])
        print(f'Episode {episode+1}/{episodes} - Reward: {ep_rewards} - Found: {found} - Coverage: {cov_pct}')
        #print epsilon
        
        coverage.append(cov_pct)
        strawberry.append(found)
        if episode % 5 == 0:
            env.write_rewards(f'results/{reward_function}/rewards_{reward_function}.csv')
            env.write_stats(f'results/{reward_function}/coverage_{reward_function}.csv',coverage_list=coverage,found_list=strawberry,reward_dict=reward_dict)
            trainer.save_models(f'models/{reward_function}/')

    
    env.write_rewards(f'results/{reward_function}/rewards_{reward_function}.csv')
    env.write_stats(f'results/{reward_function}/coverage_{reward_function}.csv',coverage_list=coverage,found_list=strawberry,reward_dict=reward_dict)
    trainer.save_models(f'models/{reward_function}/')

### --- Evaluate all agents ---

