import numpy as np
import random
from keras.activations import relu, linear
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

    def select_action(self, agent, state):
        # print(f"state shape action: {state.shape}")
        if np.random.rand() < self.epsilon:
            return self.env.action_space(agent).sample()
        state = np.expand_dims(state, axis=0) 
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

            # print(f"states shape: {states.shape}")
            # print(f"next_states shape: {next_states.shape}")
            # reshape the states to match the input shape of the model

            # usando o modelo para selecionar as melhores acoes
            next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
            
            targets = rewards + self.gamma * (next_max) * (1 - terminals)
            targets_full = self.model.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])
            
            # usando os q-valores para atualizar os pesos da rede
            # print(f"indexes: {indexes.shape} actions: {actions.shape} targets: {targets.shape}")
            # print(f"targets_full: {targets_full.shape}")
            targets_full[[indexes], [actions]] = targets

            self.model.fit(states, targets_full, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec


# receives the environment and the learners as a dictionary and fits one model for each agent
class Trainer():
    def __init__ (self, env,learners,max_steps):
        self.env = env
        self.learners = learners
        self.max_steps = max_steps


    # Train agents in the environment for AN EPISODE
    def train(self):
        steps = 0
        found = False 

        # Reset the environment and flatten the initial observations
        observations,cov_pct,found = self.env.reset()

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

            steps += 1


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
            steps += 1

        return self.rewards


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
