import gymnasium as gym
import torch
import numpy as np
import os
import ale_py
from gymnasium.wrappers import RecordVideo, ResizeObservation, FrameStackObservation, AtariPreprocessing
from networks import ImpalaCNN 

# Registrar entornos
gym.register_envs(ale_py)

# --- CONFIGURACION ---
ENV_NAME = "ALE/BasicMath-v5"

# USAMOS EL CHECKPOINT DE 10 MILLONES (EL BUENO)
MODEL_PATH = "./models/AsyncDQN_10000000.pt"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- WRAPPERS ---
class BinaryWrapper(gym.ObservationWrapper):
    def __init__(self, env, threshold=100):
        super().__init__(env)
        self.threshold = threshold
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)
    def observation(self, obs):
        return (obs > self.threshold).astype(np.float32)

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env, top=0, bottom=0, left=0, right=0):
        super().__init__(env)
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        orig_h, orig_w = env.observation_space.shape
        new_h = orig_h - top - bottom
        new_w = orig_w - left - right
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(new_h, new_w), dtype=env.observation_space.dtype)
    def observation(self, obs):
        end_y = -self.bottom if self.bottom > 0 else None
        end_x = -self.right if self.right > 0 else None
        return obs[self.top:end_y, self.left:end_x]

def make_eval_env():
    # frameskip=1 es vital para evitar conflictos
    env = gym.make(ENV_NAME, render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
    env = CropWrapper(env, top=12, bottom=8, left=0, right=0)
    env = ResizeObservation(env, (84, 84))
    env = BinaryWrapper(env, threshold=80)
    env = FrameStackObservation(env, 4)
    return env

# --- EVALUACION ---
def record_game():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: No encuentro el modelo en", MODEL_PATH)
        return

    print("Cargando modelo:", MODEL_PATH)
    
    # Crear entorno y red
    env = make_eval_env()
    # Envolver para grabar
    video_folder = "./video_final"
    env = RecordVideo(env, video_folder, disable_logger=True)
    
    # Cargar pesos
    input_shape = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = ImpalaCNN(input_channels=input_shape, depths=[16, 32, 32], output_shape=n_actions).to(DEVICE)
    
    # map_location asegura carga en CPU o GPU segun disponibilidad
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Manejo de errores de diccionario vs state_dict puro
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net.load_state_dict(checkpoint)
        
    net.eval() 

    print("Grabando partida...")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        obs_tensor = torch.tensor(np.array([obs])).to(DEVICE)
        with torch.no_grad():
            # Accion GREEDY (Sin aleatoriedad)
            action = net(obs_tensor).argmax(dim=1).item()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    print("Partida terminada! Recompensa Total:", total_reward)
    print("Video guardado en:", video_folder)

if __name__ == "__main__":
    record_game()