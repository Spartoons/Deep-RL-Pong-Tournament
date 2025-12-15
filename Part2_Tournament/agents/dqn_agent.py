import time
import datetime
import warnings
import collections
import numpy as np
import gymnasium as gym
import ale_py
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from networks import ImpalaCNN 

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
gym.register_envs(ale_py)

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
ENV_NAME = "ALE/Pong-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Params
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EXPERIENCE_REPLAY_SIZE = 10_000
SYNC_TARGET_NETWORK = 1_000

# Epsilon (Exploration) Params
EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02

# Logging Params
MEAN_REWARD_BOUND = 0
NUMBER_OF_REWARDS_TO_AVERAGE = 10

# ==========================================
# 2. Environment Setup
# ==========================================

class HorizontalFlipObservation(gym.ObservationWrapper):
    """Augment data by horizontally flipping the image."""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.flip(obs, axis=1).copy()

def make_env(env_name):
    """Creates the Pong environment with necessary wrappers."""
    env = gym.make(env_name, render_mode=None, frameskip=4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Custom and SuperSuit wrappers
    env = HorizontalFlipObservation(env)
    env = ss.color_reduction_v0(env, mode="B")        # Grayscale
    env = ss.resize_v1(env, x_size=84, y_size=84)     # Downsample
    env = ss.frame_stack_v1(env, 4, stack_dim=0)      # Stack 4 frames
    env = ss.dtype_v0(env, dtype=np.float32)          # Convert to float
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1) # Normalize 0-1
    env = ss.reshape_v0(env, (4, 84, 84))             # Channels first
    return env

# ==========================================
# 3. Experience Replay
# ==========================================

Experience = collections.namedtuple('Experience', 
    field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8), 
            np.array(next_states)
        )

# ==========================================
# 4. Agent
# ==========================================

class Agent:
    def __init__(self, env, exp_replay_buffer):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self._reset()

    def _reset(self):
        self.current_state = self.env.reset()[0]
        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device="cpu"):
        """
        Takes one step in the environment.
        Returns: The total reward if the episode ended, otherwise None.
        """
        done_reward = None

        # Epsilon-Greedy Action Selection
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([self.current_state])).to(device)
            q_vals = net(state_tensor)
            _, act_ = torch.max(q_vals, dim=1)
            action = int(act_.item())

        # Environment Step
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        # Store Experience
        exp = Experience(self.current_state, action, reward, is_done, new_state)
        self.exp_replay_buffer.append(exp)
        self.current_state = new_state

        # Handle Episode End
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

# ==========================================
# 5. Training Loop
# ==========================================

def train():
    print(f">>> Training starts at {datetime.datetime.now()}")
    
    # Initialize WandB
    wandb.login()
    wandb.init(project="PreLeague")

    # Initialize Environment and Networks
    env = make_env(ENV_NAME)
    input_shape = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = ImpalaCNN(input_channels=input_shape, depths=[8, 16, 16], output_shape=n_actions).to(DEVICE)
    target_net = ImpalaCNN(input_channels=input_shape, depths=[8, 16, 16], output_shape=n_actions).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
    agent = Agent(env, buffer)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    frame_number = 0
    epsilon = EPS_START

    while True:
        frame_number += 1
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

        # 1. Agent takes a step
        reward = agent.step(net, epsilon, device=DEVICE)

        # 2. Log progress if episode finished
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
            
            print(f"Frame: {frame_number} | Games: {len(total_rewards)} | Mean Rew: {mean_reward:.3f} | Eps: {epsilon:.2f}", flush=True)
            wandb.log({"epsilon": epsilon, "ep_mean_rew": mean_reward, "reward": reward}, step=frame_number)

            if mean_reward > MEAN_REWARD_BOUND:
                print(f"SOLVED in {frame_number} frames!")
                break

        # 3. Training Step (only if buffer has enough data)
        if len(buffer) < BATCH_SIZE: 
            continue

        # Sample batch
        states_np, actions_np, rewards_np, dones_np, next_states_np = buffer.sample(BATCH_SIZE)

        # Convert to tensors
        states = torch.tensor(states_np).to(DEVICE)
        next_states = torch.tensor(next_states_np).to(DEVICE)
        actions = torch.tensor(actions_np).to(DEVICE)
        rewards = torch.tensor(rewards_np).to(DEVICE)
        dones = torch.BoolTensor(dones_np).to(DEVICE)

        # Calculate Q Values
        Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Calculate Target Q Values
        with torch.no_grad():
            next_state_values = target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0 # No future reward if done
            expected_Q_values = next_state_values * GAMMA + rewards

        # Backpropagation
        loss = nn.MSELoss()(Q_values, expected_Q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Sync Target Network
        if frame_number % SYNC_TARGET_NETWORK == 0:
            target_net.load_state_dict(net.state_dict())

        # 5. Checkpoints
        if frame_number % 100_000 == 0:
            torch.save(net.state_dict(), f"./models/04_DQN_{frame_number}.pt")

    # Save final agent
    torch.save(agent, "./models/04_DQN_Final.pt")
    print(f">>> Training ends at {datetime.datetime.now()}")
    wandb.finish()

if __name__ == "__main__":
    train()