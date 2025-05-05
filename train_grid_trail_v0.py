import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium_env.grid_trail import GridTrailRenderEnv
from stable_baselines3 import PPO
import numpy as np
import os
import sys
import torch
import time

# Create a single-agent environment for each agent
class SingleAgentEnv(gym.Env):
    """Environment wrapper for a single agent in a multi-agent setting"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, size=10, agent_idx=0, num_agents=4, trail_lifetime=50, render_mode=None):
        super().__init__()
        self.size = size
        self.agent_idx = agent_idx  # Which agent this environment controls
        self.num_agents = num_agents
        self.trail_lifetime = trail_lifetime
        self.render_mode = render_mode
        
        # Create the actual environment
        self.env = GridTrailRenderEnv(
            size=size,
            num_agents=num_agents,
            trail_lifetime=trail_lifetime,
            render_mode=render_mode
        )
        
        # Define spaces for a single agent
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(5, 5), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        
        # Track other agents' actions
        self.other_agents_actions = [0] * (num_agents - 1)
        self.curr_observations = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        # Reset the actual multi-agent environment
        observations, info = self.env.reset(seed=seed)
        
        # Store the observations for all agents
        self.curr_observations = observations
        
        # Return only this agent's observation
        return observations[self.agent_idx], info
    
    def step(self, action):
        """Take a step for this agent only"""
        # Construct the full action list
        actions = []
        agent_counter = 0
        
        for i in range(self.num_agents):
            if i == self.agent_idx:
                actions.append(action)
            else:
                # Use the stored action for other agents
                actions.append(self.other_agents_actions[agent_counter])
                agent_counter += 1
        
        # Take the step in the multi-agent environment
        observations, reward, terminated, truncated, info = self.env.step(tuple(actions))
        
        # Store the observations
        self.curr_observations = observations
        
        # Return only this agent's observation
        return observations[self.agent_idx], reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    def set_other_actions(self, actions):
        """Set actions for other agents"""
        self.other_agents_actions = actions
    
    def get_all_observations(self):
        """Get observations for all agents"""
        return self.curr_observations

class MultiAgentTrainer:
    """Coordinator for training multiple PPO agents"""
    
    def __init__(self, size=10, num_agents=4, trail_lifetime=50, use_cuda=False):
        self.size = size
        self.num_agents = num_agents
        self.trail_lifetime = trail_lifetime
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        
        # Create individual environments for each agent
        self.envs = [
            SingleAgentEnv(
                size=size,
                agent_idx=i,
                num_agents=num_agents,
                trail_lifetime=trail_lifetime,
                render_mode=None  # No rendering during training
            ) for i in range(num_agents)
        ]
        
        # Create PPO model for each agent
        self.models = [
            PPO(
                "MlpPolicy",
                env,
                verbose=0,
                device=self.device,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.01  # Encourage exploration
            ) for i, env in enumerate(self.envs)
        ]
        
        # For visualization/testing
        self.test_env = None
    
    def train(self, total_timesteps=100000):
        """Train all agents"""
        # Create log directory
        os.makedirs("log", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize trackers
        steps_taken = 0
        episodes = 0
        episode_rewards = []
        moving_avg_reward = 0
        
        # Training loop
        while steps_taken < total_timesteps:
            # Reset all environments
            observations = []
            for env in self.envs:
                obs, _ = env.reset()
                observations.append(obs)
            
            # Initialize episode tracking
            episode_reward = 0
            done = False
            episode_steps = 0
            
            # Run episode
            while not done and steps_taken < total_timesteps:
                # Get actions from all policies
                actions = []
                for i, model in enumerate(self.models):
                    action, _ = model.predict(observations[i], deterministic=False)
                    actions.append(action)
                
                # Set other agents' actions in each environment
                for i, env in enumerate(self.envs):
                    other_actions = actions[:i] + actions[i+1:]
                    env.set_other_actions(other_actions)
                
                # Step all environments with their own action
                next_observations = []
                rewards = []
                dones = []
                infos = []
                
                for i, (env, action) in enumerate(zip(self.envs, actions)):
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_observations.append(next_obs)
                    rewards.append(reward)
                    dones.append(terminated or truncated)
                    infos.append(info)
                
                # Check for episode termination (if any agent is done)
                done = any(dones)
                
                # Use the first reward (they should all be the same)
                reward = rewards[0]
                episode_reward += reward
                
                # Update observations
                observations = next_observations
                
                # Increment counters
                steps_taken += 1
                episode_steps += 1
                
                # Periodically learn
                if steps_taken % 2048 == 0:
                    for i, model in enumerate(self.models):
                        print(f"Training agent {i} at step {steps_taken}")
                        model.learn(total_timesteps=1, reset_num_timesteps=False)
                
                # Display progress
                if steps_taken % 1000 == 0:
                    print(f"Step {steps_taken}/{total_timesteps} - Moving avg reward: {moving_avg_reward:.2f}")
            
            # Episode completed
            episodes += 1
            episode_rewards.append(episode_reward)
            
            # Update moving average
            if len(episode_rewards) > 10:
                moving_avg_reward = sum(episode_rewards[-10:]) / 10
            else:
                moving_avg_reward = sum(episode_rewards) / len(episode_rewards)
            
            print(f"Episode {episodes} completed with reward {episode_reward:.2f}, steps: {episode_steps}")
            
            # Save checkpoints every 10 episodes
            if episodes % 10 == 0:
                self.save(f"data/ppo_grid_trail_ep{episodes}")
        
        # Save final models
        self.save("data/ppo_grid_trail_final")
        print(f"Training completed after {episodes} episodes and {steps_taken} steps.")
        return episode_rewards
    
    def save(self, path_prefix):
        """Save all models"""
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(f"{path_prefix}_agent_{i}")
        print(f"Saved models to {path_prefix}")
    
    def load(self, path_prefix):
        """Load all models"""
        for i in range(self.num_agents):
            model_path = f"{path_prefix}_agent_{i}"
            if not os.path.exists(f"{model_path}.zip"):
                raise FileNotFoundError(f"Model file {model_path}.zip not found.")
            self.models[i] = PPO.load(model_path, device=self.device)
        print(f"Loaded models from {path_prefix}")
    
    def test(self, num_episodes=5, render_mode="human"):
        """Test the trained agents"""
        # Create a test environment for visualization
        test_env = GridTrailRenderEnv(
            size=self.size,
            num_agents=self.num_agents,
            trail_lifetime=self.trail_lifetime,
            render_mode=render_mode
        )
        
        for episode in range(num_episodes):
            observations, _ = test_env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Get actions from all policies
                actions = []
                for i, model in enumerate(self.models):
                    action, _ = model.predict(observations[i], deterministic=True)
                    actions.append(action)
                
                # Step the environment
                observations, reward, terminated, truncated, info = test_env.step(tuple(actions))
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                print(f"Episode {episode+1}, Step {step}, Reward: {reward}")
                
                # Add delay for visualization
                if render_mode == "human":
                    time.sleep(0.2)
            
            print(f"Episode {episode+1} completed with total reward {episode_reward}")
        
        test_env.close()

def main():
    """Main function for training or testing"""
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    # Environment parameters
    ENV_SIZE = 10
    NUM_AGENTS = 4
    TRAIL_LIFETIME = 50
    
    if mode == "train":
        # Create trainer and train
        trainer = MultiAgentTrainer(
            size=ENV_SIZE,
            num_agents=NUM_AGENTS,
            trail_lifetime=TRAIL_LIFETIME
        )
        trainer.train(total_timesteps=100000)
    
    elif mode == "test":
        # Create trainer and load models
        trainer = MultiAgentTrainer(
            size=ENV_SIZE,
            num_agents=NUM_AGENTS,
            trail_lifetime=TRAIL_LIFETIME
        )
        
        try:
            # Try to load the final model first
            trainer.load("data/ppo_grid_trail_final")
        except FileNotFoundError:
            # Fall back to the latest checkpoint
            import glob
            checkpoints = glob.glob("data/ppo_grid_trail_ep*_agent_0.zip")
            if not checkpoints:
                print("No trained models found. Please run training first with: python train_grid_trail_v0.py train")
                return
                
            latest = max(checkpoints, key=lambda x: int(x.split("ep")[1].split("_")[0]))
            latest_prefix = latest.rsplit("_agent_0.zip", 1)[0]
            print(f"Loading latest checkpoint: {latest_prefix}")
            trainer.load(latest_prefix)
            
        # Run test episodes
        trainer.test(num_episodes=5)
    
    else:
        print("Invalid mode. Use 'train' or 'test'.")

if __name__ == "__main__":
    main()