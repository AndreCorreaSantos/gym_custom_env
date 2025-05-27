
from DeepQLearning import Evaluator, build_agents
import os
from env.grid_trail import GridTrailParallelEnv

size = 40
num_agents = 5
gamma = 0.99
epsilon = 0.0  # EPSILON IS SET TO 0.0 FOR EVALUATION
epsilon_min = 0.00
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


for reward_function in ['v0', 'v1', 'v2']:


    env_params['reward'] = reward_function
    env = GridTrailParallelEnv(**env_params)

    env.reset()
    learner_params['env'] = env
    learners = build_agents(**learner_params)

    evaluator = Evaluator(env=env, learners=learners, max_steps=max_steps, max_episodes=episodes)
    evaluator.load_models(f'models/{reward_function}/')

    for episode in range(episodes):

        ep_rewards = evaluator.evaluate()
        # observations,cov_pct,found = env.reset()
        env.reset()

    # env.write_rewards(f'evaluation_results/rewards_{reward_function}.csv')
