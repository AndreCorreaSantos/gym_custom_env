# Ant Hill

## Installation steps:

```bash
python -m venv venv 
source venv/bin/activate 
pip install -r requirements.txt 
```

## Run the environment:

```bash
python run_grid_trail_v0.py
```

### Apresentação:

https://github.com/user-attachments/assets/7d4ae192-2087-4ebc-96ff-8b5254656e27


### Present the network and learning algorithm.

- premisse that this network and these hyperparameters are capable of solving the environment


### Following the network and learning algo, the reward function was messed with to try and 

### V0:

Reward function:
    - +1 for all agents if strawberry was found.
    - 0 for all agents if strawberry was not found.

Main problems observed:
    - Reward function too sparse, agents took too long to learn any useful search patterns.
    - Episode took too long to end -> because agents would only find strawberry randomly towards the end of the episode.
    - Most steps would not provide meaningful training data.

### Add training graph

### V1:

To increase the value of each agent step in its training, the reward function was adapted in order to provide more insightful feedback relating to the agent's behaviour.

Reward function:

- Global reward pool (shared between agents)
- -1 for the reward pool if an agent ran over a pheromone cell.
- +1 for the reward pool if an agent ran over a non pheromone cell.
- +100 for all agents if any of them found the strawberry.

Main problems observed:

### Add training graph

### V2:

A global reward pool was observed to provide undirected feedback to agents. That is, if one was behaving properly, all would benefit. Yet if one was misbehaving, all would be harmed. This introduces noise to the agent's learning and thus, might make the agent's training slower to converge.

To assess this hypothesis, a new training function was developed:

Reward function:

- Individual rewards
- -1 for the agent if it ran over a pheromone cell.
- +1 for the agent if it ran over a non pheromone cell.
- +100 for the agent if it found the strawberry.

### Add training graph


### Main critiques:

environment was too slow, thus, it was only possible to simulate few episodes
