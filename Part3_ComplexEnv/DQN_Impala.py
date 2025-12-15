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
import glob
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, AtariPreprocessing
import os

from networks import ImpalaCNN 

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
gym.register_envs(ale_py)

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
ENV_NAME = "ALE/BasicMath-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Params
NUM_ENVS = 16
BATCH_SIZE = 128
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
EXPERIENCE_REPLAY_SIZE = 50_000
SYNC_TARGET_NETWORK = 1_000
START_LEARNING = 10_000
TOTAL_FRAMES = 10_000_000

# Epsilon (Exploration) Params
EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02

# Logging Params
MEAN_REWARD_BOUND = 100
NUMBER_OF_REWARDS_TO_AVERAGE = 10

# ==========================================
# 2. CUSTOM WRAPPERS
# ==========================================
class BinaryWrapper(gym.ObservationWrapper):
    def __init__(self, env, threshold=100):
        super().__init__(env)
        self.threshold = threshold
        # El espacio sigue siendo el mismo, pero los valores serán extremos
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=env.observation_space.shape, 
            dtype=np.float32
        )

    def observation(self, obs):
        # Si el pixel > umbral -> 255 (Blanco), si no -> 0 (Negro)
        # BasicMath tiene fondo negro y números claros, esto resaltará los números.
        return (obs > self.threshold).astype(np.float32)

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env, top=0, bottom=0, left=0, right=0):
        super().__init__(env)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

        # Calculamos el nuevo tamaño de la observación
        orig_h, orig_w = env.observation_space.shape
        new_h = orig_h - top - bottom
        new_w = orig_w - left - right
        
        # Actualizamos el espacio de observación para que Gym no se queje
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(new_h, new_w), dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        # Slicing de numpy: [Y_inicio:Y_fin, X_inicio:X_fin]
        # Si bottom es 0, vamos hasta el final, si no, restamos
        end_y = -self.bottom if self.bottom > 0 else None
        end_x = -self.right if self.right > 0 else None
        
        return obs[self.top:end_y, self.left:end_x]
    
# ==========================================
# 3. Environment Factory (Async Logic)
# ==========================================

def make_env_config(env, seed):
    """Aplica la configuración de Wrappers (compartida entre Train y Eval)"""
    # 1. Preprocesamiento Atari básico (manteniendo tamaño original)
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False
    )
    # 2. Custom Crop
    env = CropWrapper(env, top=12, bottom=8, left=0, right=0)
    # 3. Resize
    env = ResizeObservation(env, (84, 84))
    # 4. Binary
    env = BinaryWrapper(env, threshold=80)
    # 5. Stack
    env = FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def make_train_env(env_name, seed):
    def thunk():
        env = gym.make(env_name, frameskip=1)
        env = make_env_config(env, seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ==========================================
# 4. Classes (Buffer & Agent) - Optimizadas
# ==========================================
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self): return len(self.buffer)
    def append(self, experience): self.buffer.append(experience)
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states))

class VectorAgent:
    def __init__(self, envs, exp_replay_buffer):
        self.envs = envs
        self.exp_replay_buffer = exp_replay_buffer
        self.num_envs = envs.num_envs
        self.current_states, _ = self.envs.reset()

    def step(self, net, epsilon=0.0, device="cpu", global_step=0):
        if np.random.random() < epsilon:
            actions = self.envs.action_space.sample() 
        else:
            state_tensor = torch.tensor(self.current_states).to(device)
            with torch.no_grad():
                q_vals = net(state_tensor)
                _, act_ = torch.max(q_vals, dim=1)
                actions = act_.cpu().numpy()

        new_states, rewards, terminated, truncated, infos = self.envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        for i in range(self.num_envs):
            exp = Experience(self.current_states[i], actions[i], rewards[i], dones[i], new_states[i])
            self.exp_replay_buffer.append(exp)
            
            if "_episode" in infos and infos["_episode"][i]:
                r = infos["episode"]["r"][i]
                wandb.log({"ep_mean_rew": r, "epsilon": epsilon}, step=global_step)

        self.current_states = new_states
        return self.num_envs

# ==========================================
# 5. Training Loop
# ==========================================
def train():
    print(f">>> Training starts at {datetime.datetime.now()}")

    os.makedirs("./models", exist_ok=True)
    
    wandb.login()
    wandb.init(project="Project_PML_Part3", name="DQN_Async_NoVideo")

    # 1. Crear Entornos Async
    envs = gym.vector.AsyncVectorEnv([
        make_train_env(ENV_NAME, seed=42+i) for i in range(NUM_ENVS)
    ])

    input_shape = envs.single_observation_space.shape[0] 
    n_actions = envs.single_action_space.n
    print(f"Async Envs: {NUM_ENVS} | Input: {envs.single_observation_space.shape}")

    net = ImpalaCNN(input_channels=input_shape, depths=[16, 32, 32], output_shape=n_actions).to(DEVICE)
    target_net = ImpalaCNN(input_channels=input_shape, depths=[16, 32, 32], output_shape=n_actions).to(DEVICE)
    target_net.load_state_dict(net.state_dict())

    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
    agent = VectorAgent(envs, buffer)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    frame_number = 0
    epsilon = EPS_START

    while frame_number < TOTAL_FRAMES:
        frames_added = agent.step(net, epsilon, device=DEVICE, global_step=frame_number)
        frame_number += frames_added
        
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

        if len(buffer) < START_LEARNING: continue

        if frame_number % 4 == 0: 
            states_np, actions_np, rewards_np, dones_np, next_states_np = buffer.sample(BATCH_SIZE)
            states = torch.tensor(states_np).to(DEVICE)
            next_states = torch.tensor(next_states_np).to(DEVICE)
            actions = torch.tensor(actions_np).to(DEVICE)
            rewards = torch.tensor(rewards_np).to(DEVICE)
            dones = torch.BoolTensor(dones_np).to(DEVICE)

            Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_state_values = target_net(next_states).max(1)[0]
                next_state_values[dones] = 0.0 
                expected_Q_values = next_state_values * GAMMA + rewards

            loss = nn.MSELoss()(Q_values, expected_Q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if frame_number % SYNC_TARGET_NETWORK < NUM_ENVS:
            target_net.load_state_dict(net.state_dict())
            
        if frame_number % 500_000 < NUM_ENVS:
            torch.save(net.state_dict(), f"./models/AsyncDQN_{frame_number}.pt")

    envs.close()
    torch.save(agent, "./models/AsyncDQN_Final.pt")
    wandb.finish()

if __name__ == "__main__":
    train()