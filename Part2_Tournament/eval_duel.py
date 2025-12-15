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
import supersuit as ss
from pettingzoo.atari import pong_v3

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
    "save_frequency": 50_000,
    "video_frequency": 500_000,
    "project_name": "Pong-League-Arena",
    "load_model_path_left": "/fhome/pmlai08/SlaveLeague/LEAGUE/pong_league_arena/league_models/beta.pt",
    "load_model_path_right": "/fhome/pmlai08/SlaveLeague/LEAGUE/pong_league_arena/league_models/alpha.pt"
}

config["batch_size"] = int(config["num_envs"] * config["num_steps"])
config["minibatch_size"] = int(config["batch_size"] // config["num_minibatches"])

# --- 2. ENVIRONMENT FACTORIES ---
def make_env():
    env = pong_v3.parallel_env(render_mode=None)
    
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(env, num_vec_envs=16, num_cpus=5, base_class='gymnasium')
    return env

def make_eval_env(env_id, run_name, step, capture_video=True):
    video_folder = os.path.join(BASE_DIR, f"videos/{run_name}/step_{step}")
    os.makedirs(video_folder, exist_ok=True)
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder, disable_logger=True)
    else:
        env = gym.make(env_id)
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

# --- 6. MODULAR FUNCTIONS ---
# --- 1. Setup & Initialization ---
def setup_run(config, run_name):
    """Initializes WandB, directories, seeds, and device."""
    print(f"--> STARTING PPG TRAINING ({run_name})...", flush=True)
    run = wandb.init(project=config["project_name"], config=config, monitor_gym=False, save_code=True)
    
    os.makedirs(os.path.join(BASE_DIR, f"models/{run.id}"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, f"videos/{run.id}"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Device: {device}", flush=True)

    # Seeding
    # random.seed(config["seed"]) # Import random if needed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = config["seed"] == 0
    
    return run, device

def create_optimizer(agent, config):
    return optim.Adam(agent.parameters(), lr=config["learning_rate"], eps=1e-5)

# --- 1. ROLLOUT COLLECTION (1v1 Logic) ---
def collect_rollout_1v1(envs, agent_left, agent_right, next_obs, next_done, config, device, global_step, window_rewards_left, window_rewards_right):
    """
    Collects experience by letting agent_left and agent_right play against each other.
    Assumes envs is a SuperSuit vectorized env where:
    - Even indices (0, 2, 4...) are Player 1 (Left)
    - Odd indices (1, 3, 5...) are Player 2 (Right)
    """
    # Calculate storage shapes based on single agent view (half the batch size)
    # However, we store the full batch to simplify stepping, and slice later.
    total_batch_size = config["num_envs"] * 2 # 16 envs * 2 agents = 32
    obs_shape = next_obs.shape[1:]
    act_shape = envs.action_space.shape[1:]

    # Buffers (Store everything, we will filter later)
    obs_buf = torch.zeros((config["num_steps"], total_batch_size) + obs_shape).to(device)
    actions_buf = torch.zeros((config["num_steps"], total_batch_size) + act_shape).to(device)
    logprobs_buf = torch.zeros((config["num_steps"], total_batch_size)).to(device)
    rewards_buf = torch.zeros((config["num_steps"], total_batch_size)).to(device)
    dones_buf = torch.zeros((config["num_steps"], total_batch_size)).to(device)
    values_buf = torch.zeros((config["num_steps"], total_batch_size)).to(device)

    for step in range(config["num_steps"]):
        global_step += config["num_envs"] # Count distinct environments steps
        obs_buf[step] = next_obs
        dones_buf[step] = next_done

        # --- 1. Split Observations ---
        # Evens = Left Agent, Odds = Right Agent
        obs_left = next_obs[0::2]
        obs_right = next_obs[1::2]

        # --- 2. Get Actions from Both Agents ---
        with torch.no_grad():
            # Left Agent
            act_l, logp_l, _, val_l = agent_left.get_action_and_value(obs_left)
            # Right Agent
            act_r, logp_r, _, val_r = agent_right.get_action_and_value(obs_right)

        # --- 3. Interleave Actions for the Environment ---
        # We need to reconstruct the full batch of actions [L, R, L, R...]
        actions = torch.zeros((total_batch_size,), dtype=act_l.dtype).to(device)
        logprobs = torch.zeros((total_batch_size,), dtype=logp_l.dtype).to(device)
        values = torch.zeros((total_batch_size,), dtype=val_l.dtype).to(device)

        actions[0::2] = act_l
        actions[1::2] = act_r
        logprobs[0::2] = logp_l
        logprobs[1::2] = logp_r
        values[0::2] = val_l.flatten()
        values[1::2] = val_r.flatten()

        # Store
        actions_buf[step] = actions
        logprobs_buf[step] = logprobs
        values_buf[step] = values

        # --- 4. Step Environment ---
        next_obs_np, reward, terminations, truncations, infos = envs.step(actions.cpu().numpy())
        next_done_np = np.logical_or(terminations, truncations)

        rewards_buf[step] = torch.tensor(reward).to(device).view(-1)
        next_obs = torch.Tensor(next_obs_np).to(device)
        next_done = torch.Tensor(next_done_np).to(device)

        # Logging (Standard PettingZoo/SuperSuit info structure)
        if "episode" in infos:
            # Info is also vectorized [L, R, L, R...]
            for i in range(total_batch_size):
                if infos["_episode"][i]:
                    r = infos["episode"]["r"][i]
                    if isinstance(r, (np.ndarray, torch.Tensor)): r = r.item()
                    
                    # Split Rewards by Agent
                    if i % 2 == 0:
                        # Even index = Left Agent
                        window_rewards_left.append(r)
                    else:
                        # Odd index = Right Agent
                        window_rewards_right.append(r)

    buffer_data = {
        "obs": obs_buf, "actions": actions_buf, "logprobs": logprobs_buf,
        "rewards": rewards_buf, "dones": dones_buf, "values": values_buf,
        "next_obs": next_obs, "next_done": next_done, "global_step": global_step
    }
    return buffer_data

# --- 2. DATA FILTERING ---
def filter_batch_for_agent(buffer_data, agent_idx_offset):
    """
    Slices the buffer to get only the data relevant for the specific agent.
    agent_idx_offset: 0 for Left Agent (Evens), 1 for Right Agent (Odds).
    """
    filtered = {}
    for k, v in buffer_data.items():
        # Check specific keys FIRST to handle dimensions correctly
        if k in ["next_obs", "next_done"]: 
            # These are (num_envs_total, ...). We want to slice the 0th dimension (Envs).
            filtered[k] = v[agent_idx_offset::2]
        elif isinstance(v, torch.Tensor) and v.ndim >= 2:
            # These are (num_steps, num_envs_total, ...). We want to slice the 1st dimension (Envs).
            filtered[k] = v[:, agent_idx_offset::2] 
        else:
            filtered[k] = v # scalars like global_step
    return filtered

# --- 3. GAE COMPUTATION ---
def compute_gae(agent, buffer_data, config, device):
    """Calculates advantages. Input buffer_data should be filtered for the specific agent already."""
    rewards = buffer_data["rewards"]
    dones = buffer_data["dones"]
    values = buffer_data["values"]
    next_obs = buffer_data["next_obs"]
    next_done = buffer_data["next_done"]
    num_steps = rewards.shape[0]

    with torch.no_grad():
        next_val = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
        returns = advantages + values
    
    return advantages, returns

# --- 4. PPO UPDATE ---
def train_policy_phase(agent, optimizer, buffer_data, advantages, returns, envs, config, aux_memory):
    """Updates the policy using PPO and stores data for PPG Aux phase."""
    
    # buffer_data["obs"] shape is (num_steps, num_envs_subset, C, H, W)
    # We want to flatten the first two dimensions (steps * envs)
    obs_shape = buffer_data["obs"].shape[2:]
    b_obs = buffer_data["obs"].reshape((-1,) + obs_shape)
    
    # Same for actions
    act_shape = buffer_data["actions"].shape[2:]
    b_actions = buffer_data["actions"].reshape((-1,) + act_shape)

    b_logprobs = buffer_data["logprobs"].reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    # -- PPG: Store for Aux --
    with torch.no_grad():
        out = agent.actor(b_obs)
        current_logits = out[0] if isinstance(out, tuple) else out
    
    aux_memory["obs"].append(b_obs.clone().cpu())
    aux_memory["returns"].append(b_returns.clone().cpu())
    aux_memory["logits"].append(current_logits.clone().cpu())

    # -- PPO Optimization --
    current_batch_size = b_obs.shape[0] 
    b_inds = np.arange(current_batch_size)
    
    for epoch in range(config["update_epochs"]):
        np.random.shuffle(b_inds)
        for start in range(0, current_batch_size, config["minibatch_size"]):
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
            
    return loss.item(), v_loss.item()

# --- 5. AUXILIARY PHASE ---
def train_aux_phase(agent, optimizer, aux_memory, config, device):
    """Standard PPG Aux phase."""
    if len(aux_memory["obs"]) == 0: return

    concat_obs = torch.cat(aux_memory["obs"])
    concat_returns = torch.cat(aux_memory["returns"])
    concat_old_logits = torch.cat(aux_memory["logits"])
    
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

    # Clear memory
    aux_memory["obs"] = []
    aux_memory["returns"] = []
    aux_memory["logits"] = []

# --- 7. MAIN TRAINING LOOP ---
def train(config):
    # Setup
    run, device = setup_run(config, "placeholder")
    
    # Environment (PettingZoo Vectorized)
    envs = make_env() 
    
    # Initialize Agents
    agent_left = PPGAgent(envs, device).to(device)
    optimizer_left = create_optimizer(agent_left, config)
    
    agent_right = PPGAgent(envs, device).to(device)
    optimizer_right = create_optimizer(agent_right, config)

    # Load Models
    if os.path.exists(config["load_model_path_left"]):
        agent_left.load_state_dict(torch.load(config["load_model_path_left"], map_location=device, weights_only=True), strict=False)
        print("--> Loaded Left Agent")
    if os.path.exists(config["load_model_path_right"]):
        agent_right.load_state_dict(torch.load(config["load_model_path_right"], map_location=device, weights_only=True), strict=False)
        print("--> Loaded Right Agent")

    # Tracking
    aux_mem_left = {"obs": [], "returns": [], "logits": []}
    aux_mem_right = {"obs": [], "returns": [], "logits": []}
    
    # Init Environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config["num_envs"] * 2).to(device) # Note *2 for both agents
    
    # --- CHANGED: Separate Reward Windows ---
    window_rewards_left = collections.deque(maxlen=100)
    window_rewards_right = collections.deque(maxlen=100)

    num_updates = config["total_timesteps"] // config["batch_size"]
    update = 0

    while True:
        update += 1
        
        # Determine who we are training this cycle (or both)
        # For simplicity, let's say we alternate updates, but we ALWAYS collect data for both.
        train_left_now = ((update//100) % 2 != 0) # Train left on odd updates
        train_right_now = ((update//100) % 2 == 0) # Train right on even updates

        # 1. COLLECT ROLLOUT (Both agents play)
        # --- CHANGED: Pass both windows ---
        full_batch = collect_rollout_1v1(
            envs, agent_left, agent_right, next_obs, next_done, 
            config, device, global_step, window_rewards_left, window_rewards_right
        )
        
        # Update trackers
        next_obs = full_batch["next_obs"]
        next_done = full_batch["next_done"]
        global_step = full_batch["global_step"]

        # 2. TRAIN LEFT AGENT
        if train_left_now:
            # Extract only Even indices (Left agent's view)
            batch_left = filter_batch_for_agent(full_batch, agent_idx_offset=0)
            
            # GAE
            adv_l, ret_l = compute_gae(agent_left, batch_left, config, device)
            
            # Policy Update
            pl, vl = train_policy_phase(
                agent_left, optimizer_left, batch_left, adv_l, ret_l, 
                envs, config, aux_mem_left
            )
            
            # Aux Update
            if update % config["policy_phases"] == 0:
                print(f"--> AUX PHASE (LEFT) at step {global_step}")
                train_aux_phase(agent_left, optimizer_left, aux_mem_left, config, device)

        # 3. TRAIN RIGHT AGENT
        if train_right_now:
            # Extract only Odd indices (Right agent's view)
            batch_right = filter_batch_for_agent(full_batch, agent_idx_offset=1)
            
            # GAE
            adv_r, ret_r = compute_gae(agent_right, batch_right, config, device)
            
            # Policy Update
            pl, vl = train_policy_phase(
                agent_right, optimizer_right, batch_right, adv_r, ret_r, 
                envs, config, aux_mem_right
            )
            
            # Aux Update
            if update % config["policy_phases"] == 0:
                print(f"--> AUX PHASE (RIGHT) at step {global_step}")
                train_aux_phase(agent_right, optimizer_right, aux_mem_right, config, device)

        # Logging and Saving (Standard)
        if update % 10 == 0:
            # --- CHANGED: Calculate means separately ---
            mean_rew_left = np.mean(window_rewards_left) if len(window_rewards_left) > 0 else 0
            mean_rew_right = np.mean(window_rewards_right) if len(window_rewards_right) > 0 else 0
            
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step} | Left: {mean_rew_left:.2f} | Right: {mean_rew_right:.2f} | SPS: {sps}")
            
            # --- CHANGED: Log separately ---
            wandb.log({
                "ep_mean_rew_left": mean_rew_left,
                "ep_mean_rew_right": mean_rew_right,
                "global_step": global_step, 
                "SPS": sps
            })

        if global_step % config["save_frequency"] < config["batch_size"]:
             torch.save(agent_left.state_dict(), f"{BASE_DIR}/models/{run.id}/left_{global_step}.pt")
             torch.save(agent_right.state_dict(), f"{BASE_DIR}/models/{run.id}/right_{global_step}.pt")
        
        if global_step >= config["total_timesteps"]:
            break

    envs.close()
    wandb.finish()

if __name__ == "__main__":
    train(config)