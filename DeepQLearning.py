import numpy as np
import random
from keras.activations import relu, linear
import gc
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
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

    def train(self):
        done = {agent: False for agent in self.env.agents}
        steps = 0
    

        # Reset the environment and flatten the initial observations
        observations = self.env.reset()
        observations = {
            agent: observations[agent]
            for agent in self.env.agents
        }

        while not any(done.values()) and steps < self.max_steps:
            # Save current state before step

            # Select actions for each agent using reshaped input

            actions = {
                agent: self.learners[agent].select_action(
                    agent,
                    observations[agent]
                )
                for agent in self.env.agents
            }

            # Step the environment
            observations, rewards, terminations, truncations, infos = self.env.step(actions) # overwrite observations

            observations = {
                agent:  observations[agent] 
                for agent in self.env.agents
            }

            # Store experience and train
            for agent in self.env.agents:
                # print(f"training agent: {agent}")
                self.learners[agent].experience(
                    observations[agent],   # correct old state
                    actions[agent],
                    rewards[agent],
                    observations[agent],        # correct new state
                    terminations[agent]
                )
                self.learners[agent].experience_replay()

            print(f"steps: {steps}")

            done = {
                agent: terminations[agent] or truncations[agent]
                for agent in self.env.agents
            }
            steps += 1

        # Only needed if you're hitting memory issues
        keras.backend.clear_session()
        gc.collect()
        self.env.close()

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
    
    def evaluate(self):
        for episode in range(self.max_episodes):
            done = {agent: False for agent in self.env.agents}
            steps = 0
            observations = self.env.reset()
            observations = {
                agent: observations[agent]
                for agent in self.env.agents
            }
            while not any(done.values()) and steps < self.max_steps:
                actions = {
                    agent: self.learners[agent].select_action(
                        agent,
                        observations[agent]
                    )
                    for agent in self.env.agents
                }
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                observations = {
                    agent:  observations[agent] 
                    for agent in self.env.agents
                }
                for agent in self.env.agents:
                    self.rewards[agent].append(rewards[agent])
                done = {
                    agent: terminations[agent] or truncations[agent]
                    for agent in self.env.agents
                }
                steps += 1

        return self.rewards

def build_model(input_dim, output_dim, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model