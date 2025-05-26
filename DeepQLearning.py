import numpy as np
import random
from keras.activations import relu, linear
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import psutil
import os

from collections import deque



class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec,  batch_size, memory, model):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.training_steps = 0  # Track training steps for cleanup

    def select_action(self, agent, state):
        # print(f"state shape action: {state.shape}")
        if np.random.rand() < self.epsilon:
            return self.env.action_space(agent).sample()
        
        state = np.expand_dims(state, axis=0) 
        # Use predict with explicit cleanup
        with keras.utils.custom_object_scope({}):
            action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        # soh acontece o treinamento depois da memoria ser maior que o batch_size informado
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            terminals = np.array([i[4] for i in batch])

            # np.squeeze(): Remove single-dimensional entries from the shape of an array.
            # Para se adequar ao input
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # usando o modelo para selecionar as melhores acoes
            next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
            
            targets = rewards + self.gamma * (next_max) * (1 - terminals)
            targets_full = self.model.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])
            
            # usando os q-valores para atualizar os pesos da rede
            targets_full[[indexes], [actions]] = targets

            self.model.fit(states, targets_full, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

            # Increment training steps and cleanup periodically
            self.training_steps += 1
            
            # Cleanup every 100 training steps
            if self.training_steps % 100 == 0:
                self.cleanup_memory()
            
            # Explicit cleanup of batch variables
            del batch, states, actions, rewards, next_states, terminals
            del targets, targets_full, indexes, next_max

    def cleanup_memory(self):
        """Clean up memory to prevent leaks"""
        # Force garbage collection
        gc.collect()
        
        # Clear Keras backend session every 500 training steps to prevent graph buildup
        if self.training_steps % 500 == 0:
            print(f"[Agent] Clearing Keras session at training step {self.training_steps}")
            keras.backend.clear_session()
            gc.collect()


# receives the environment and the learners as a dictionary and fits one model for each agent
class Trainer():
    def __init__ (self, env,learners,max_steps):
        self.env = env
        self.learners = learners
        self.max_steps = max_steps
        self.episode_count = 0
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    # Train agents in the environment for AN EPISODE
    def train(self):
        steps = 0
        found = False 
        
        # Track memory usage
        if self.initial_memory is None:
            self.initial_memory = self.get_memory_usage()
        
        current_memory = self.get_memory_usage()
        memory_growth = current_memory - self.initial_memory

        # Reset the environment and flatten the initial observations
        observations,cov_pct,found = self.env.reset()

        # sum of the rewards for each agent for the episode
        reward_dict = {}
        for agent in self.env.agents:
            reward_dict[agent] = 0

        observations = {
            agent: observations[agent]
            for agent in self.env.agents
        }

        while not found and steps < self.max_steps:

            # Select actions for each agent using reshaped input
            actions = {
                agent: self.learners[agent].select_action(
                    agent,
                    observations[agent]
                )
                for agent in self.env.agents
            }

            # Step the environment
            observations, rewards, found = self.env.step(actions) # overwrite observations

            observations = {
                agent:  observations[agent] 
                for agent in self.env.agents
            }
            # terminal state if the agent found the goal
            terminal = found 
            # Store experience and train
            for agent in self.env.agents:
                # print(f"training agent: {agent}")
                self.learners[agent].experience(
                    observations[agent],   # correct old state
                    actions[agent],
                    rewards[agent],
                    observations[agent],        # correct new state
                    terminal
                )
                self.learners[agent].experience_replay()

            # print(f"steps: {steps}")

            for agent in self.env.agents:
                reward_dict[agent] += rewards[agent]

            steps += 1
            
            # Memory monitoring every 100 steps
            if steps % 100 == 0:
                current_memory = self.get_memory_usage()
                memory_growth = current_memory - self.initial_memory
                print(f"Step {steps}: Memory usage: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
                
                # Force aggressive cleanup if memory grows too much
                if memory_growth > 1000:  # More than 1GB growth
                    print(f"WARNING: High memory usage detected. Forcing cleanup...")
                    self.force_cleanup()

        # Episode-level cleanup
        self.episode_count += 1
        
        # Clean up episode variables
        del observations, actions, rewards
        
        # Periodic episode-level cleanup
        if self.episode_count % 10 == 0:
            print(f"Episode {self.episode_count}: Performing periodic cleanup")
            current_memory = self.get_memory_usage()
            memory_growth = current_memory - self.initial_memory
            print(f"Memory usage: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
            
            for agent in self.env.agents:
                self.learners[agent].cleanup_memory()
            
            gc.collect()

        return reward_dict

    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        print("Performing aggressive memory cleanup...")
        
        for agent in self.env.agents:
            # Force cleanup for all agents
            self.learners[agent].cleanup_memory()
        
        # Clear Keras session
        keras.backend.clear_session()
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Reset memory baseline
        self.initial_memory = self.get_memory_usage()
        print(f"Cleanup complete. New memory baseline: {self.initial_memory:.1f}MB")

    # Save models for each agent on folder at path
    def save_models(self,path):
        for agent in self.env.agents:
            self.learners[agent].model.save(f"{path}{agent}.keras")
            print(f"Model for {agent} saved at {path}{agent}.keras")


class Evaluator():
    def __init__(self, env, learners, max_steps,max_episodes):
        self.max_episodes = max_episodes
        self.env = env
        self.learners = learners
        self.rewards = {agent: [] for agent in env.agents}
        self.max_steps = max_steps
    
    def load_models(self, path):
        for agent in self.env.agents:
            self.learners[agent].model = keras.models.load_model(f"{path}{agent}.keras")
            print(f"Model for {agent} loaded from {path}{agent}.keras")

    # Evaluate agents in the environment for one episode 
    def evaluate(self):
        steps = 0
        observations, cov_pct, found = self.env.reset() 

        observations = {
            agent: observations[agent]
            for agent in self.env.agents
        }
        reward_dict = {}
        for agent in self.env.agents:
            reward_dict[agent] = 0

        while not found and steps < self.max_steps:
            actions = {
                agent: self.learners[agent].select_action(
                    agent,
                    observations[agent]
                )
                for agent in self.env.agents
            }
            observations, rewards, found = self.env.step(actions)

            observations = {
                agent: observations[agent]
                for agent in self.env.agents
            }

            if self.env.render_mode == "human":
                self.env.render()

            for agent in self.env.agents:
                self.rewards[agent].append(rewards[agent])
                reward_dict[agent] += rewards[agent]
            steps += 1

        # Cleanup after evaluation episode
        del observations, actions, rewards
        gc.collect()

        return reward_dict


def build_model(input_dim, output_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


def build_agents(env, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,batch_size=64, memory_size=20000):
    sample_agent = env.agents[0]
    input_dim = np.prod(env.observation_space(sample_agent).shape)
    n_actions = env.action_space(sample_agent).n

    learners = {}
    for agent in env.agents:
        model = build_model(input_dim=input_dim, output_dim=n_actions)
        memory = deque(maxlen=memory_size)
        learners[agent] = DeepQLearning(
            env=env,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_dec=epsilon_decay,
            batch_size=batch_size,
            memory=memory,
            model=model
        )

    return learners