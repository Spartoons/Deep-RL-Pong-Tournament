# -*- coding: utf-8 -*-
import os
import random
import time
import collections
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
import ale_py
# RecordEpisodeStatistics es vital para las graficas
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, AtariPreprocessing, RecordEpisodeStatistics

# --- IMPORT IMPALA NETWORK ---
try:
    from networks import ImpalaCNN
except ImportError:
    print("CRITICAL ERROR: 'networks.py' not found.")
    exit()

gym.register_envs(ale_py)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. CONFIGURATION ---
config = {
    "env_name": "ALE/BasicMath-v5",
    "seed": 42,
    "total_timesteps": 100_000_000,
    "num_steps": 256,
    "num_envs": 16,
    "num_minibatches": 8,
    "update_epochs": 1,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 5e-4,
    "clip_coef": 0.2,
    "anneal_lr": True,
    "norm_adv": True,
    "policy_phases": 16,
    "aux_epochs": 6,
    "aux_minibatch_size": 1024,
    "kl_coef": 1.0,
    "save_frequency": 500_000,
    # "video_frequency": Eliminado
    "project_name": "Project_PML_Part3",
    "load_model_path": None
}

config["batch_size"] = int(config["num_envs"] * config["num_steps"])
config["minibatch_size"] = int(config["batch_size"] // config["num_minibatches"])

# --- 2. CUSTOM WRAPPERS (Keep consistent with DQN) ---  
class TimePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=-0.005):
        super().__init__(env)
        self.penalty = penalty
    def reward(self, reward):
        if reward > 0: return 1.0
        return self.penalty


class ActivityRewardWrapper(gym.Wrapper):
    def __init__(self, env, bonus=0.001):
        super().__init__(env)
        self.bonus = bonus
        self.last_obs = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calcular diferencia con el frame anterior (movimiento visual)
        if self.last_obs is not None:
            # Simple diferencia de pixeles (L1 norm)
            diff = np.abs(obs - self.last_obs).mean()
            
            # Si la pantalla cambia algo (se mueve), dar miguita de pan
            if diff > 0.01: 
                reward += self.bonus
        
        self.last_obs = obs.copy()
        return obs, reward, terminated, truncated, info
            
class BinaryWrapper(gym.ObservationWrapper):
    def __init__(self, env, threshold=100):
        super().__init__(env)
        self.threshold = threshold
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return (obs > self.threshold).astype(np.float32)

class CropWrapper(gym.ObservationWrapper):
    def __init__(self, env, top=0, bottom=0, left=0, right=0):
        super().__init__(env)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
        orig_h, orig_w = env.observation_space.shape
        new_h = orig_h - top - bottom
        new_w = orig_w - left - right
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(new_h, new_w), dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        end_y = -self.bottom if self.bottom > 0 else None
        end_x = -self.right if self.right > 0 else None
        return obs[self.top:end_y, self.left:end_x]

# --- 3. ENVIRONMENT FACTORY ---
def make_env_config(env, seed):
    """Shared config"""
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84, 
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False
    )
    env = CropWrapper(env, top=12, bottom=8, left=0, right=0)
    env = ResizeObservation(env, (84, 84))
    env = BinaryWrapper(env, threshold=80)
    env = TimePenaltyWrapper(env, penalty=-0.005)
    env = ActivityRewardWrapper(env, bonus=0.01)
    env = FrameStackObservation(env, 4)
    
    
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id, frameskip=1)
        env = make_env_config(env, seed)
        env = RecordEpisodeStatistics(env)
        return env
    return thunk

# --- 4. HELPER FOR HOOKS ---
class CaptureFeatures:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = input[0]
    
    def close(self):
        self.hook.remove()

# --- 5. PPG AGENT ARCHITECTURE ---
class PPGAgent(nn.Module):
    def __init__(self, envs, device, action_dim=6, input_channels=4):
        super().__init__()
        
        self.actor = ImpalaCNN(
            input_channels=input_channels,
            depths=[16, 32, 32],
            output_shape=action_dim
        )
        
        self.critic = ImpalaCNN(
            input_channels=input_channels,
            depths=[16, 32, 32],
            output_shape=1
        )

        modules = list(self.actor.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linear_layers) > 0:
            self.actor_feature_hook = CaptureFeatures(linear_layers[-1])
            feature_dim = linear_layers[-1].in_features
        else:
            raise ValueError("Linear layer not found in ImpalaCNN.")

        self.aux_critic = nn.Linear(feature_dim, 1)

        torch.nn.init.orthogonal_(self.aux_critic.weight, 1.0)
        torch.nn.init.constant_(self.aux_critic.bias, 0.0)

    def get_value(self, x):
        out = self.critic(x)
        if isinstance(out, tuple): return out[0]
        return out

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if isinstance(logits, tuple): logits = logits[0]

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        value = self.get_value(x)
        return action, probs.log_prob(action), probs.entropy(), value

    def get_aux_values(self, x):
        logits = self.actor(x)
        if isinstance(logits, tuple): logits = logits[0]
        
        features = self.actor_feature_hook.features
        aux_value = self.aux_critic(features)
        return logits, aux_value

# --- 6. MAIN TRAINING LOOP ---
def train():
    print("--> STARTING PPG TRAINING (FAST TRACK)...", flush=True)

    run = wandb.init(project=config["project_name"], config=config, monitor_gym=False, save_code=True, name="PPG_Impala_Fast")
    
    # Create only models folder
    os.makedirs(os.path.join(BASE_DIR, f"models/{run.id}"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}", flush=True)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["seed"] == 0

    envs = gym.vector.AsyncVectorEnv(
        [make_env(config["env_name"], config["seed"] + i, i, run.id) for i in range(config["num_envs"])]
    )
    
    action_dim = envs.single_action_space.n
    agent = PPGAgent(envs, device, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    if config.get("load_model_path") and os.path.exists(config["load_model_path"]):
        checkpoint = torch.load(config["load_model_path"], map_location=device)
        agent.load_state_dict(checkpoint)
        print("--> Model loaded.")

    obs = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    rewards = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    dones = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    values = torch.zeros((config["num_steps"], config["num_envs"])).to(device)

    aux_obs_buffer = []
    aux_returns_buffer = []
    aux_logits_buffer = []

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config["num_envs"]).to(device)
    
    num_updates = config["total_timesteps"] // config["batch_size"]
    window_rewards = collections.deque(maxlen=100)
    
    update = 0

    while global_step < config["total_timesteps"]:
        update += 1
        
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        # --- ROLLOUT ---
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "episode" in infos and "_episode" in infos:
                for i in range(config["num_envs"]):
                    if infos["_episode"][i]:
                        r = infos["episode"]["r"][i]
                        wandb.log({"episodic_return": r, "global_step": global_step})
                        window_rewards.append(r)

        # --- GAE & UPDATE ---
        with torch.no_grad():
            next_val = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        with torch.no_grad():
             out = agent.actor(b_obs)
             current_logits = out[0] if isinstance(out, tuple) else out
        
        aux_obs_buffer.append(b_obs.clone().cpu())
        aux_returns_buffer.append(b_returns.clone().cpu())
        aux_logits_buffer.append(current_logits.clone().cpu())

        b_inds = np.arange(config["batch_size"])

        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = agent.get_value(b_obs[mb_inds]).view(-1)
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + config["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

        if update % config["policy_phases"] == 0:
            concat_obs = torch.cat(aux_obs_buffer)
            concat_returns = torch.cat(aux_returns_buffer)
            concat_old_logits = torch.cat(aux_logits_buffer)
            aux_inds = np.arange(concat_obs.shape[0])
            
            for _ in range(config["aux_epochs"]):
                np.random.shuffle(aux_inds)
                for start in range(0, len(aux_inds), config["aux_minibatch_size"]):
                    end = start + config["aux_minibatch_size"]
                    mb_inds = aux_inds[start:end]
                    
                    mb_obs = concat_obs[mb_inds].to(device)
                    mb_returns = concat_returns[mb_inds].to(device)
                    mb_old_logits = concat_old_logits[mb_inds].to(device)

                    new_logits, new_aux_value = agent.get_aux_values(mb_obs)
                    new_aux_value = new_aux_value.view(-1)
                    aux_loss = 0.5 * ((new_aux_value - mb_returns) ** 2).mean()
                    probs_old = Categorical(logits=mb_old_logits)
                    probs_new = Categorical(logits=new_logits)
                    kl_loss = torch.distributions.kl_divergence(probs_old, probs_new).mean()
                    joint_loss = aux_loss + config["kl_coef"] * kl_loss
                    
                    optimizer.zero_grad()
                    joint_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                    optimizer.step()
            
            aux_obs_buffer = []
            aux_returns_buffer = []
            aux_logits_buffer = []

        if update % 10 == 0:
            mean_rew = np.mean(window_rewards) if len(window_rewards) > 0 else 0
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step} | Return: {mean_rew:.2f} | SPS: {sps}", flush=True)
            wandb.log({
                "ep_mean_rew": mean_rew,
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "charts/SPS": sps,
                "global_step": global_step
            })

        # Save only models, no video
        if global_step % config["save_frequency"] < config["batch_size"]:
            save_path = os.path.join(BASE_DIR, f"models/{run.id}/model_{global_step}.pth")
            torch.save(agent.state_dict(), save_path)

    final_path = os.path.join(BASE_DIR, f"models/{run.id}/model_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"--> Final model saved to {final_path}")
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    train()