import os
import random
import time
import glob
import multiprocessing
import collections
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordEpisodeStatistics, RecordVideo

# --- CUSTOM IMPORT ---
# We assume this works. The code below fixes the "return value" issue via hooks.
try:
    from networks import ImpalaCNN
except ImportError:
    print("WARNING: 'networks.py' not found. Creating a Mock ImpalaCNN for testing.")
    class ImpalaCNN(nn.Module):
        def __init__(self, input_channels, depths, output_shape):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(input_channels, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Flatten()
            )
            self.final = nn.Linear(64 * 9 * 9, output_shape)
        def forward(self, x):
            x = self.backbone(x)
            return self.final(x)

# Register Atari
gym.register_envs(ale_py)

PARALLEL_AGENTS = 8
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. CONFIGURATION ---
config = {
    "env_name": "PongNoFrameskip-v4",
    "seed": 1,
    "total_timesteps": 7_500_000,
    "num_steps": 128,
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
    "save_frequency": 1_000_000,
    "video_frequency": 500_000,
    "project_name": "PreLeague",
    "load_model_path": "/fhome/pmlai08/SlaveLeague/LEAGUE/preleague/models/khxmo1nb/model_1001472.pth"
}

config["batch_size"] = int(config["num_envs"] * config["num_steps"])
config["minibatch_size"] = int(config["batch_size"] // config["num_minibatches"])

# --- 2. ENVIRONMENT FACTORIES ---
class HorizontalFlipObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    def observation(self, obs):
        return np.flip(obs, axis=1).copy()


def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = HorizontalFlipObservation(env)
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
        env = FrameStackObservation(env, 4)
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_eval_env(env_id, run_name, step, capture_video=True):
    video_folder = os.path.join(BASE_DIR, f"videos/{run_name}/step_{step}")
    os.makedirs(video_folder, exist_ok=True)
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder, disable_logger=True)
    else:
        env = gym.make(env_id)
    env = HorizontalFlipObservation(env)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, 4)
    return env

# --- 3. HELPER FOR HOOKS ---
class CaptureFeatures:
    """Hooks into a layer to capture its input (features) during forward pass."""
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # input is a tuple (x,), we want x
        self.features = input[0]
    
    def close(self):
        self.hook.remove()

# --- 4. PPG AGENT ARCHITECTURE ---
class PPGAgent(nn.Module):
    def __init__(self, envs, device, action_dim=6, input_channels=4):
        super().__init__()
        
        # 1. The POLICY Network (Actor)
        self.actor = ImpalaCNN(
            input_channels=input_channels,
            depths=[32, 64, 64],
            output_shape=action_dim
        )
        
        # 2. The VALUE Network (True Critic)
        self.critic = ImpalaCNN(
            input_channels=input_channels,
            depths=[32, 64, 64],
            output_shape=1
        )

        # 3. Setup Feature Extraction Hook for Actor
        # We need to find the final Linear layer to hook its input.
        # This assumes the last module in the network is the head.
        modules = list(self.actor.modules())
        # Filter for Linear layers
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        if len(linear_layers) > 0:
            # We assume the last linear layer is the output head
            self.actor_feature_hook = CaptureFeatures(linear_layers[-1])
            feature_dim = linear_layers[-1].in_features
        else:
            raise ValueError("Could not find a Linear layer in ImpalaCNN to hook features from.")

        # 4. Auxiliary Value Head
        self.aux_critic = nn.Linear(feature_dim, 1)

        # Init aux head
        torch.nn.init.orthogonal_(self.aux_critic.weight, 1.0)
        torch.nn.init.constant_(self.aux_critic.bias, 0.0)

    def get_value(self, x):
        """Returns value from the TRUE Critic."""
        # Handle cases where Critic returns (val, feat) or just val
        out = self.critic(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def get_action_and_value(self, x, action=None):
        """
        Returns action distribution and value (from True Critic).
        """
        # Run Actor (Hook captures features automatically)
        logits = self.actor(x)
        if isinstance(logits, tuple): logits = logits[0] # Handle if it actually returns tuple

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        # Run True Critic
        value = self.get_value(x)
        
        return action, probs.log_prob(action), probs.entropy(), value

    def get_aux_values(self, x):
        """
        Returns (logits, aux_value) from the POLICY network.
        """
        # Run Actor
        logits = self.actor(x)
        if isinstance(logits, tuple): logits = logits[0]
        
        # Retrieve captured features from the hook
        features = self.actor_feature_hook.features
        
        # Run Aux Head
        aux_value = self.aux_critic(features)
        
        return logits, aux_value

# --- 5. EVALUATION AND RECORDING ---
def evaluate_and_record(agent, device, run_name, step):
    """
    Runs one episode in a separate environment to capture video.
    """
    print(f"--> EVALUATION AT STEP {step}...", flush=True)
    eval_env = make_eval_env(config["env_name"], run_name, step=step, capture_video=True)
    
    # Reset
    next_obs, _ = eval_env.reset()
    next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device) # Add batch dim
    
    done = False
    total_reward = 0
    
    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs)
        
        next_obs, reward, terminated, truncated, info = eval_env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        next_obs = torch.Tensor(next_obs).unsqueeze(0).to(device)
    
    eval_env.close()
    
    # Find the video file generated by RecordVideo
    video_folder = os.path.join(BASE_DIR, f"videos/{run_name}/step_{step}")
    # Look for .mp4 files created recently
    mp4_files = glob.glob(f"{video_folder}/*.mp4")
    
    wandb_dict = {"eval/episodic_return": total_reward, "global_step": step}
    
    if mp4_files:
        # Take the last created file
        latest_video = max(mp4_files, key=os.path.getctime)
        wandb_dict["eval/video"] = wandb.Video(latest_video, fps=30, format="mp4")
        print(f"    Saved Video: {latest_video}")
    
    wandb.log(wandb_dict)
    print(f"    Eval Return: {total_reward}", flush=True)

# --- 6. MAIN TRAINING LOOP ---
def train():
    print("--> STARTING PPG TRAINING...", flush=True)

    run = wandb.init(project=config["project_name"], config=config, monitor_gym=False, save_code=True)
    
    os.makedirs(os.path.join(BASE_DIR, f"models/{run.id}"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, f"videos/{run.id}"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}", flush=True)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["seed"] == 0

    envs = gym.vector.AsyncVectorEnv(
        [make_env(config["env_name"], config["seed"] + i, i, run.id) for i in range(config["num_envs"])]
    )
    
    agent = PPGAgent(envs, device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    if config.get("load_model_path") and os.path.exists(config["load_model_path"]):
        print(f"--> Loading checkpoint from: {config['load_model_path']}", flush=True)
        # map_location ensures it loads to the correct device (cpu vs cuda)
        checkpoint = torch.load(config["load_model_path"], map_location=device)
        
        # Load the weights into the agent
        agent.load_state_dict(checkpoint)
        print("--> Model weights loaded successfully.", flush=True)
    elif config.get("load_model_path"):
        print(f"--> WARNING: Model path specified but file not found: {config['load_model_path']}", flush=True)

    # Standard Buffers
    obs = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    rewards = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    dones = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    values = torch.zeros((config["num_steps"], config["num_envs"])).to(device)

    # AUXILIARY BUFFERS
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

    while True:
        update += 1
        
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        # --- POLICY PHASE: ROLLOUT ---
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

            if "episode" in infos:
                for i in range(config["num_envs"]):
                    if infos["_episode"][i]:
                        r = infos["episode"]["r"][i]
                        if isinstance(r, (np.ndarray, torch.Tensor)): r = r.item()
                        window_rewards.append(r)
                        wandb.log({"episodic_return": r, "global_step": global_step})

        # --- GAE ---
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

        # --- UPDATE ---
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # Capture current logits for KL in Aux Phase
        with torch.no_grad():
             current_logits, _ = agent.actor(b_obs) if isinstance(agent.actor(b_obs), tuple) else (agent.actor(b_obs), None)
        
        # Store for Aux Phase (Detached!)
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

        # --- AUXILIARY PHASE ---
        if update % config["policy_phases"] == 0:
            print(f"--> AUXILIARY PHASE at step {global_step}", flush=True)
            
            # Move accumulated data to GPU
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
            
            # Clear Buffer
            aux_obs_buffer = []
            aux_returns_buffer = []
            aux_logits_buffer = []

        # --- LOGGING ---
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
            if mean_rew > 0:
                break

        if global_step % config["save_frequency"] < config["batch_size"]:
            save_path = os.path.join(BASE_DIR, f"models/{run.id}/model_{global_step}.pth")
            torch.save(agent.state_dict(), save_path)

        if global_step % config["video_frequency"] < config["batch_size"]:
            evaluate_and_record(agent, device, run.id, global_step)

    final_path = os.path.join(BASE_DIR, f"models/{run.id}/model_final.pt")
    torch.save(agent.state_dict(), final_path)
    print(f"--> Final model saved to {final_path}")

    envs.close()
    wandb.finish()

if __name__ == "__main__":
    train()