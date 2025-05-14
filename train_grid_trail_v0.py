import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from env.grid_trail import GridTrailParallelEnv
from DeepQLearning import DeepQLearning, Trainer  

# --- Parameters ---
size = 40
num_agents = 5  # use 50 if your system can handle it
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 500
batch_size = 64
memory_size = 20000
max_steps = 500
learning_rate = 0.001

# --- Environment ---
env = GridTrailParallelEnv(render_mode=None, size=size, num_agents=num_agents)
env.reset()

# Sample agent name
sample_agent = env.agents[0]
input_dim = np.prod(env.observation_space(sample_agent).shape)
print(f"Observation space shape: {input_dim}")
n_actions = env.action_space(sample_agent).n

# --- Function to create a model ---
def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

# --- Create a learner per agent ---
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
        episodes=episodes,
        batch_size=batch_size,
        memory=memory,
        model=model,
        max_steps=max_steps
    )

# --- Train all agents ---
trainer = Trainer(env=env, learners=learners)

for episode in range(episodes):
    print(f"\n--- Episode {episode+1}/{episodes} ---")
    print(f"reward: {0}")
    trainer.train()
