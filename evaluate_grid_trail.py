from DeepQLearning import Evaluator, build_agents
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


env_params = {
    'render_mode': "human",
    'size': size,
    'num_agents': num_agents,
    'flatten_observations': True
}

learner_params = {
    'gamma': gamma,
    'epsilon': epsilon,
    'epsilon_min': epsilon_min,
    'epsilon_decay': epsilon_decay,
    'batch_size': batch_size,
    'memory_size': memory_size 
}

os.makedirs('evaluation_results', exist_ok=True)

for reward_function in ['v0', 'v1', 'v2']:

    os.makedirs(f'evaluation_results/{reward_function}/', exist_ok=True)

    env_params['reward'] = reward_function
    env = GridTrailParallelEnv(**env_params)
    env.reset()
    learner_params['env'] = env
    learners = build_agents(**learner_params)

    evaluator = Evaluator(env=env, learners=learners, max_steps=max_steps, max_episodes=episodes)
    evaluator.load_models(f'models/{reward_function}/')

    
    coverage = []
    strawberry = []
    for episode in range(episodes):
        evaluator.evaluate()
        observations,cov_pct,found = env.reset()
        coverage.append(cov_pct)
        strawberry.append(found)
        if episode % 10 == 0:
            print(f"Episode {episode+1}/{episodes} - Coverage: {cov_pct}, Found: {found}")
            env.write_rewards(f'evaluation_results/{reward_function}/rewards_{reward_function}.csv')
            env.write_coverage(f'evaluation_results/{reward_function}/coverage_{reward_function}.csv',coverage_list=coverage)


    # env.write_rewards(f'evaluation_results/rewards_{reward_function}.csv')
