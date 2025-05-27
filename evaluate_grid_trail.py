from DeepQLearning import Evaluator, build_agents
import os
from env.grid_trail import GridTrailParallelEnv


size = 40
num_agents = 5
gamma = 0.99
epsilon = 0.0 # EPSILON IS SET TO 0.0 FOR EVALUATION
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 5
batch_size = 64
memory_size = 20000
max_steps = 100


env_params = {
    'render_mode': None,
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
    reward_dict = {}
    for agent in env.agents:
        reward_dict[agent] = []
    for episode in range(episodes):

        ep_rewards,found,cov_pct = evaluator.evaluate()

        for agent in env.agents:
            reward_dict[agent].append(ep_rewards[agent])
        coverage.append(cov_pct)
        strawberry.append(found)

        for agent in env.agents:
            reward_dict[agent].append(ep_rewards[agent])

        print(f"Episode {episode+1}/{episodes} - Coverage: {cov_pct}, Found: {found}")
        if episode % 10 == 0:
            env.write_rewards(f'evaluation_results/{reward_function}/rewards_{reward_function}.csv')
            env.write_stats(f'evaluation_results/{reward_function}/coverage_{reward_function}.csv',coverage_list=coverage,found_list=strawberry,reward_dict=reward_dict)


    # env.write_rewards(f'evaluation_results/rewards_{reward_function}.csv')
