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

# Register Atari
gym.register_envs(ale_py)

PARALLEL_AGENTS = 8  # Adjusted for efficiency, can be lowered if CPU is weak
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_cpu_allocation():
    total_cores = multiprocessing.cpu_count()
    available_per_agent = (total_cores - 2) // PARALLEL_AGENTS
    return min(32, max(4, available_per_agent))

# --- 1. CONFIGURATION ---
config = {
    "env_name": "PongNoFrameskip-v4",
    "seed": 1,
    "total_timesteps": 7_500_000,
    "num_steps": 128,        # As requested
    "num_envs": 8,           # Number of parallel processes
    "num_minibatches": 4,
    "update_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.015,       # As requested
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 0.0008, # As requested
    "clip_coef": 0.18,       # As requested (clip_range)
    "anneal_lr": True,
    "norm_adv": True,
    "save_frequency": 1_000_000, # Save model every 1M steps
    "video_frequency": 500_000,  # Record video every 500k steps
    "project_name": "Pong_PPO_PureTorch"
}

config["batch_size"] = int(config["num_envs"] * config["num_steps"])
config["minibatch_size"] = int(config["batch_size"] // config["num_minibatches"])

# --- 2. ENVIRONMENT FACTORIES ---

def make_env(env_id, seed, idx, run_name):
    """
    Standard training environment: Optimized, Grayscale, No Video.
    """
    def thunk():
        env = gym.make(env_id)
        env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
        env = FrameStackObservation(env, 4)
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def make_eval_env(env_id, run_name, step, capture_video=True):
    """
    Evaluation environment: Records video, single instance.
    """
    video_folder = os.path.join(BASE_DIR, f"videos/{run_name}/step_{step}")
    
    os.makedirs(video_folder, exist_ok=True)
    
    if capture_video:
        # We use render_mode rgb_array for the video recorder
        env = gym.make(env_id, render_mode="rgb_array")
        # RecordVideo must wrap before pre-processing if we want color video
        env = RecordVideo(env, video_folder, disable_logger=True)
    else:
        env = gym.make(env_id)

    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, 4)
    return env

# --- 3. NEURAL NETWORK (Unchanged) ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# --- 4. EVALUATION & VIDEO FUNCTION ---
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

# --- 5. MAIN TRAINING LOOP ---
def train():
    print("--> STARTING TRAINING...", flush=True)
    
    # Initialize WandB
    run = wandb.init(
        project=config["project_name"],
        config=config,
        monitor_gym=False,
        save_code=True
    )

    wandb.define_metric("global_step")
    wandb.define_metric("ep_rew_mean", step_metric="global_step")
    
    # Create Directories
    os.makedirs(os.path.join(BASE_DIR, f"models/{run.id}"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, f"videos/{run.id}"), exist_ok=True)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}", flush=True)

    # Seeding
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["seed"] == 0

    # Vector Envs
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config["env_name"], config["seed"] + i, i, run.id) for i in range(config["num_envs"])]
    )
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

    # Storage setup
    obs = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config["num_steps"], config["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    rewards = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    dones = torch.zeros((config["num_steps"], config["num_envs"])).to(device)
    values = torch.zeros((config["num_steps"], config["num_envs"])).to(device)

    # Start tracking
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config["num_envs"]).to(device)
    num_updates = config["total_timesteps"] // config["batch_size"]
    
    window_rewards = collections.deque(maxlen=100)

    for update in range(1, num_updates + 1):
        
        # Annealing the rate if instructed to do so.
        if config["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow

        # --- COLLECT ROLLOUTS ---
        for step in range(config["num_steps"]):
            global_step += 1 * config["num_envs"]
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
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "episode" in infos:
                for i in range(config["num_envs"]):
                    if infos["_episode"][i]:
                        r = infos["episode"]["r"][i]
                        # Convert to python float if it's a tensor/array
                        if isinstance(r, (np.ndarray, torch.Tensor)): r = r.item()
                        window_rewards.append(r)
                        wandb.log({"episodic_return": r, "global_step": global_step})

        # --- GAE ESTIMATION ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # --- OPTIMIZATION ---
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + config["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

        # --- LOGGING & SAVING ---
        if update % 10 == 0:
            mean_rew = np.mean(window_rewards) if len(window_rewards) > 0 else 0
            print(f"Step {global_step} | Return: {mean_rew:.2f} | SPS: {int(global_step / (time.time() - start_time))}", flush=True)
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "ep_rew_mean": mean_rew,
                "global_step": global_step
            })

        # Save Checkpoint
        if global_step % config["save_frequency"] < config["batch_size"]:
            save_path = os.path.join(BASE_DIR, f"models/{run.id}/model_{global_step}.pth")
            torch.save(agent.state_dict(), save_path)
            print(f"--> Checkpoint saved: {save_path}")

        # Video Recording
        if global_step % config["video_frequency"] < config["batch_size"]:
            evaluate_and_record(agent, device, run.id, global_step)

    # --- FINAL SAVE ---
    final_path = os.path.join(BASE_DIR, f"models/{run.id}/model_final.pth")
    torch.save(agent.state_dict(), final_path)
    print(f"--> Final model saved to {final_path}")
    
    envs.close()
    wandb.finish()

if __name__ == "__main__":
    train()