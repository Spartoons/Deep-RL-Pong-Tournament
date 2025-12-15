import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import numpy as np
import ale_py
import time
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import wandb

# --- Hyperparameters ---
ENV_NAME = 'PongNoFrameskip-v4'
TOTAL_TIMESTEPS = 4_000_000  
NUM_ENVS = 4                 
GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EXPERIENCE_REPLAY_SIZE = 50000
SYNC_TARGET_NETWORK = 1000
EPS_START = 1.0
EPS_DECAY_STEPS = 1_000_000  
EPS_MIN = 0.02

# --- WandB Setup ---
wandb.init(
    project="pong-dqn-async",
    config={
        "env_name": ENV_NAME,
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_envs": NUM_ENVS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE
    }
)

# --- Wrappers ---
class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(old_shape[-1], old_shape[0], old_shape[1]), 
            dtype=np.float32
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = MaxAndSkipObservation(env, skip=4)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = ImageToPyTorch(env)
        env = ReshapeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)
        env = ScaledFloatFrame(env)
        return env
    return thunk

# --- Model ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def make_DQN(input_shape, output_shape):
    net = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape)
    )
    return net

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        indices = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

# --- Main Training Loop ---
print(">>> Training starts at ", datetime.datetime.now())

# Create Async Vector Environment
envs = gym.vector.AsyncVectorEnv([make_env(ENV_NAME) for _ in range(NUM_ENVS)])

print(f"Observation Space: {envs.single_observation_space.shape}")
print(f"Action Space: {envs.single_action_space.n}")

net = make_DQN(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
target_net = make_DQN(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
target_net.load_state_dict(net.state_dict())

buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Initialize States
current_states, _ = envs.reset()

# Tracking variables
running_rewards = np.zeros(NUM_ENVS)
total_rewards = []
global_step = 0
start_time = time.time()

try:
    while global_step < TOTAL_TIMESTEPS:
        # Calculate Epsilon
        fraction = min(1.0, float(global_step) / EPS_DECAY_STEPS)
        epsilon = EPS_START + fraction * (EPS_MIN - EPS_START)

        # 1. Select Actions (Batched)
        if np.random.random() < epsilon:
            actions = envs.action_space.sample()
        else:
            state_tensor = torch.tensor(current_states).to(device)
            q_vals = net(state_tensor)
            _, act_ = torch.max(q_vals, dim=1)
            actions = act_.cpu().numpy()

        # 2. Step Environments
        new_states, rewards, terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        # 3. Update Manual Reward Tracker
        running_rewards += rewards

        # 4. Handle Done Environments & Logging
        for i in range(NUM_ENVS):
            if dones[i]:
                ep_rew = running_rewards[i]
                total_rewards.append(ep_rew)
                
                # Calculate mean of last 10 games
                ep_rew_mean = np.mean(total_rewards[-10:])
                
                print(f"Frame: {global_step} | Last Rew: {ep_rew:.1f} | Mean Rew: {ep_rew_mean:.1f} | Eps: {epsilon:.3f}")
                wandb.log({"ep_mean_rew": ep_rew_mean, "ep_rew": ep_rew, "global_step": global_step})
                
                # Reset tracking for this specific environment
                running_rewards[i] = 0

            # 5. Store Experience
            exp = Experience(current_states[i], actions[i], rewards[i], dones[i], new_states[i])
            buffer.append(exp)

        # Update current states and step count
        current_states = new_states
        global_step += NUM_ENVS

        # 6. Training Step
        if len(buffer) < 1000:
            continue

        batch = buffer.sample(BATCH_SIZE)
        states_, actions_, rewards_, dones_, next_states_ = batch

        states = torch.tensor(states_).to(device)
        next_states = torch.tensor(next_states_).to(device)
        actions = torch.tensor(actions_).to(device)
        rewards = torch.tensor(rewards_).to(device)
        dones = torch.BoolTensor(dones_).to(device)

        Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            expected_Q_values = next_state_values * GAMMA + rewards

        loss = nn.MSELoss()(Q_values, expected_Q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. Sync & Periodic Status Log
        if global_step % SYNC_TARGET_NETWORK < NUM_ENVS:
            target_net.load_state_dict(net.state_dict())

        if global_step % 5000 < NUM_ENVS:
            sps = int(global_step / (time.time() - start_time))
            print(f"Status: Step {global_step} | SPS: {sps} | Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item(), "SPS": sps}, commit=False)

except KeyboardInterrupt:
    print("Training interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    raise e

# Cleanup
torch.save(net.state_dict(), f"Pong_DQN_Async_Final_{global_step}.dat")
envs.close()
wandb.finish()
print(">>> Training ends at ", datetime.datetime.now())