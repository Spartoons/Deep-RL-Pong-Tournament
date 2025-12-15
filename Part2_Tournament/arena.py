import os
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import supersuit as ss
from pettingzoo.atari import pong_v3
import wandb

# =============================================================================
# 1. IMPORTS & CONFIGURATION
# =============================================================================

# --- Custom Networks Imports ---
try:
    from a_08_DQN_I import ImpalaCNN as ImpalaCNN_DQN
    from networks import NatureCNN   
except ImportError as e:
    print(f"Warning: {e}. Using Mocks for testing.", flush=True)
    class NatureCNN(nn.Module):
        def __init__(self, input_channels, depths, output_shape): super().__init__(); self.l=nn.Linear(input_channels*84*84, output_shape)
        def forward(self, x): return self.l(x.flatten(1))
    ImpalaCNN_DQN = NatureCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LEAGUE SETTINGS ---
WANDB_PROJECT = "Pong-League-Arena"
NUM_ENVS = 16            
ROLLOUT_STEPS = 128      
TOTAL_ROUNDS = 5000      # Total training loops
EVAL_INTERVAL = 500       # Run evaluation tournament every X rounds
SAVE_INTERVAL = 100      
MODELS_DIR = "./league_models/"

# Hyperparameters
LR_DQN = 1e-4
LR_PPO = 2.5e-4
LR_PPG = 5e-4
GAMMA = 0.99

os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# 2. AGENT INTERFACE
# =============================================================================

class LeagueAgent:
    def __init__(self, name, model, optimizer):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.steps = 0
        self.elo = 1000.0  # Kept for simple tracking
        
    def get_action_batch(self, obs_tensor, eval=False): raise NotImplementedError
    def observe_batch(self, obs, actions, metadata, rewards, dones, next_obs): raise NotImplementedError
    def update(self): raise NotImplementedError
    
    def save(self):
        path = os.path.join(MODELS_DIR, f"{self.name}.pt")
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
            print(f"Loaded {self.name} from {path}", flush=True)
        else:
            print(f"Warning: {path} not found. Starting fresh.", flush=True)

# --- DQN Agent ---
class DQNAgent(LeagueAgent):
    def __init__(self, name, model_cls, input_shape, action_dim, load_path=None, depths=[8, 16, 16]):
        model = model_cls(input_channels=input_shape[0], depths=depths, output_shape=action_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR_DQN)
        super().__init__(name, model, optimizer)
        
        self.target_model = model_cls(input_channels=input_shape[0], depths=depths, output_shape=action_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = collections.deque(maxlen=50000)
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.batch_size = 64
        self.target_sync_freq = 1000
        if load_path: self.load(load_path)

    def get_action_batch(self, obs_tensor, eval=False):
        with torch.no_grad():
            q_vals = self.model(obs_tensor)
            actions = torch.argmax(q_vals, dim=1)
        
        if not eval:
            mask = torch.rand(actions.shape) < self.epsilon
            random_actions = torch.randint(0, 6, actions.shape)
            actions = torch.where(mask.to(DEVICE), random_actions.to(DEVICE), actions)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return actions, None

    def observe_batch(self, obs, actions, metadata, rewards, dones, next_obs):
        # Store on CPU RAM
        obs_np, act_np = obs.cpu().numpy(), actions.cpu().numpy()
        rew_np, done_np = rewards.cpu().numpy(), dones.cpu().numpy()
        next_obs_np = next_obs.cpu().numpy()
        for i in range(len(obs_np)):
            self.buffer.append((obs_np[i], act_np[i], rew_np[i], done_np[i], next_obs_np[i]))

    def update(self, next_obs_batch, next_done_batch):
        if len(self.buffer) < 2000: return None
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)
        
        states = torch.tensor(np.array(states)).to(DEVICE)
        actions = torch.tensor(np.array(actions)).to(DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(np.array(dones), dtype=torch.uint8).to(DEVICE)
        next_states = torch.tensor(np.array(next_states)).to(DEVICE)
        
        q_vals = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            expected_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.MSELoss()(q_vals, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_sync_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

# --- PPG Helper Classes ---
class StandalonePPGAgent(nn.Module):
    def __init__(self, model_cls, input_shape, action_dim):
        super().__init__()
        self.actor = model_cls(input_channels=input_shape[0], depths=[32,64,64], output_shape=action_dim)
        self.critic = model_cls(input_channels=input_shape[0], depths=[32,64,64], output_shape=1)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        if isinstance(logits, tuple): logits = logits[0]
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        val = self.critic(x)
        if isinstance(val, tuple): val = val[0]
        return action, probs.log_prob(action), probs.entropy(), val

class PPGAgentWrapper(LeagueAgent):
    def __init__(self, name, model_cls, input_shape, action_dim, load_path=None):
        model = StandalonePPGAgent(model_cls, input_shape, action_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR_PPG)
        super().__init__(name, model, optimizer)
        self.rollout_buffer = [] 
        if load_path: self.load(load_path)
        
    def get_action_batch(self, obs_tensor, eval=False):
        with torch.no_grad():
            action, logprob, _, value = self.model.get_action_and_value(obs_tensor)
        # If eval, we might want deterministic (argmax) instead of sampling, 
        # but standard PPO/PPG eval often keeps sampling. Let's keep sampling for consistency.
        return action, {"logprob": logprob, "value": value}

    def observe_batch(self, obs, actions, metadata, rewards, dones, next_obs):
        self.rollout_buffer.append({
            "obs": obs, "action": actions, "logprob": metadata["logprob"],
            "value": metadata["value"].flatten(), "reward": rewards, "done": dones
        })

    def update(self, next_obs_batch=None, next_done_batch=None):
            if not self.rollout_buffer: return None

            # 1. Prepare Batch Data
            # Flatten the buffer list into tensors
            b_obs = torch.cat([x["obs"] for x in self.rollout_buffer])
            b_act = torch.cat([x["action"] for x in self.rollout_buffer])
            b_logprobs = torch.cat([x["logprob"] for x in self.rollout_buffer])
            
            # Stack scalar values
            val_steps = torch.stack([x["value"] for x in self.rollout_buffer])
            rew_steps = torch.stack([x["reward"] for x in self.rollout_buffer])
            done_steps = torch.stack([x["done"] for x in self.rollout_buffer])

            # 2. Bootstrapping (GAE Calculation)
            with torch.no_grad():
                if next_obs_batch is not None:
                    # Get value of the state AFTER the last step (Step 129)
                    _, _, _, next_value = self.model.get_action_and_value(next_obs_batch)
                    next_value = next_value.reshape(-1)
                else:
                    next_value = torch.zeros_like(val_steps[0])

            advantages = torch.zeros_like(rew_steps)
            lastgaelam = 0
            for t in reversed(range(len(rew_steps))):
                if t == len(rew_steps) - 1:
                    nextnonterminal = 1.0 - next_done_batch.float() if next_done_batch is not None else 1.0
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_steps[t]
                    nextvalues = val_steps[t+1]
                
                delta = rew_steps[t] + GAMMA * nextvalues * nextnonterminal - val_steps[t]
                advantages[t] = lastgaelam = delta + GAMMA * 0.95 * nextnonterminal * lastgaelam

            returns = advantages + val_steps
            
            # Flatten for the update loop
            b_returns = returns.reshape(-1)
            b_adv = advantages.reshape(-1)

            # 3. Advantage Normalization (Perform ONCE on the full batch, not inside minibatch)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            # 4. Optimization Loop
            b_inds = np.arange(len(b_obs))
            np.random.shuffle(b_inds)
            
            total_loss = 0
            updates = 0
            
            # PPO Hyperparameters
            CLIP_COEF = 0.2
            ENT_COEF = 0.01
            VF_COEF = 0.5
            MAX_GRAD_NORM = 0.5

            for start in range(0, len(b_obs), 128):
                end = start + 128
                mb_inds = b_inds[start:end]

                # Forward pass on minibatch
                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                    b_obs[mb_inds], 
                    b_act[mb_inds]
                )

                logratio = (newlogprob - b_logprobs[mb_inds]).exp()
                mb_adv = b_adv[mb_inds]

                # Policy Loss (Minimize negative objective)
                # We want to MAXIMIZE the objective, so we MINIMIZE the negative
                pg_loss1 = -mb_adv * logratio
                pg_loss2 = -mb_adv * torch.clamp(logratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Clipped pattern is optional, simpler MSE here is standard for PPG/PPO)
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss (Negative because we want to maximize entropy)
                entropy_loss = entropy.mean()
                
                loss = pg_loss - (ENT_COEF * entropy_loss) + (VF_COEF * v_loss)

                self.optimizer.zero_grad()
                loss.backward()
                
                # CRITICAL: Clip gradients to prevent instability
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                
                self.optimizer.step()
                total_loss += loss.item()
                updates += 1

            self.rollout_buffer = []
            return total_loss / updates if updates > 0 else 0

# =============================================================================
# 3. EVALUATION LOGIC
# =============================================================================

def run_evaluation_tournament(left_agents, right_agents, round_idx):
    """
    Runs a deterministic round of matches to gauge performance.
    Records video of the first active match in the batch.
    """
    print(">>> STARTING EVALUATION TOURNAMENT (Video Enabled)...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
    
    # 1. Create a specific EVAL environment with rendering enabled
    # We use fewer parallel envs for eval if we want to save memory, 
    # but using the same NUM_ENVS is fine if VRAM allows.
    eval_env = pong_v3.parallel_env(render_mode="rgb_array")
    eval_env = ss.color_reduction_v0(eval_env, mode="B")
    eval_env = ss.resize_v1(eval_env, x_size=84, y_size=84)
    eval_env = ss.frame_stack_v1(eval_env, 4, stack_dim=0)
    eval_env = ss.dtype_v0(eval_env, dtype=np.float32)
    eval_env = ss.normalize_obs_v0(eval_env, env_min=0, env_max=1)
    eval_env = ss.reshape_v0(eval_env, (4, 84, 84))
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    
    # Force 1 CPU for eval to avoid rendering conflicts
    eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=NUM_ENVS, num_cpus=0, base_class='gymnasium')

    results = collections.defaultdict(lambda: {"wins": 0, "games": 0, "total_reward": 0})
    
    # Generate Pairs
    l_keys = list(left_agents.keys())
    r_keys = list(right_agents.keys())
    all_pairs = []
    for l_name in l_keys:
        for r_name in r_keys:
            all_pairs.append((l_name, r_name))
    
    frames = [] # To store video frames
    
    # Chunk pairs
    for i in range(0, len(all_pairs), NUM_ENVS):
        batch_pairs = all_pairs[i : i + NUM_ENVS]
        active_count = len(batch_pairs)
        
        # Padding
        while len(batch_pairs) < NUM_ENVS:
            batch_pairs.append(batch_pairs[0])
            
        obs, _ = eval_env.reset()
        
        # Run for longer (10000 steps) to ensure points are scored
        for step in range(10000):
            actions = np.zeros(NUM_ENVS * 2, dtype=int)
            
            # Inference
            for env_idx, (pL, pR) in enumerate(batch_pairs):
                # Left
                obs_L = torch.tensor(obs[env_idx*2]).unsqueeze(0).to(DEVICE)
                act_L, _ = left_agents[pL].get_action_batch(obs_L, eval=True)
                actions[env_idx*2] = act_L.item()
                

                obs_R = torch.tensor(obs[env_idx*2]).unsqueeze(0).to(DEVICE)
                act_R, _ = right_agents[pR].get_action_batch(obs_R, eval=True)
                actions[env_idx*2+1] = act_R.item()
            
            # Step
            next_obs, rewards, term, trunc, _ = eval_env.step(actions)
            
            # --- VIDEO CAPTURE ---
            # We only capture the first batch of matches to save time/space
            if i == 0 and step % 2 == 0: # Save every 2nd frame to save size
                # eval_env.render() returns the big grid of all envs. 
                # We just want one, but Supersuit/VecEnv renders them concatenated usually.
                # Let's try to get the full render.
                try:
                    frame = eval_env.render() 
                    frames.append(frame)
                except Exception as e:
                    print(f"Render Error: {e}")

            # Stats
            for env_idx in range(active_count):
                pL, pR = batch_pairs[env_idx]
                results[pL]["total_reward"] += rewards[env_idx*2]
                results[pR]["total_reward"] += rewards[env_idx*2+1]
                
                if term[env_idx*2] or trunc[env_idx*2]:
                    results[pL]["games"] += 1
                    results[pR]["games"] += 1
                    rL = rewards[env_idx*2]
                    if rL > 0: results[pL]["wins"] += 1
                    if rL < 0: results[pR]["wins"] += 1
            
            obs = next_obs
            
            # Early break if all games in this batch finished? 
            # Hard to track with auto-reset, so fixed steps is safer for now.

    eval_env.close()

    # --- LOGGING ---
    print("--- EVAL RESULTS ---")
    log_data = {}
    for name, stats in results.items():
        win_rate = stats["wins"] / stats["games"] if stats["games"] > 0 else 0
        avg_rew = stats["total_reward"] / stats["games"] if stats["games"] > 0 else 0
        print(f"{name}: Win Rate {win_rate:.2f} | Avg Rew {avg_rew:.2f}")
        log_data[f"eval/{name}_win_rate"] = win_rate
        log_data[f"eval/{name}_avg_reward"] = avg_rew

    # Process Video
    if len(frames) > 0:
        # WandB expects (Time, Channel, Height, Width)
        # Gym render is usually (Height, Width, Channel)
        frames = np.array(frames)
        # Transpose to (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        log_data["eval/match_video"] = wandb.Video(frames, fps=30, format="mp4", caption=f"Round {round_idx} Eval")

    wandb.log(log_data)

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

def make_env():
    env = pong_v3.parallel_env(render_mode=None)
    
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    env = ss.concat_vec_envs_v1(env, num_vec_envs=NUM_ENVS, num_cpus=5, base_class='gymnasium')
    return env

def main():
    wandb.init(project=WANDB_PROJECT, name="10-Warrior-League")

    print(">>> Initializing Environment (CPU)...", flush=True)
    env = make_env()
    obs, _ = env.reset()
    print(">>> Environment Ready.", flush=True)

    # 1. Load 10 Agents (5 Left, 5 Right)
    # Note: I am assuming file paths. Update them to match your exact filenames.
    
    left_agents = {}
    right_agents = {}

    print(">>> Loading Warriors to GPU...", flush=True)
    
    # --- Loading Left Side (Example Names) ---
    # left_agents["b_PPG"] = PPGAgentWrapper("b_PPG", NatureCNN, (4,84,84), 6, "./league_models/b_PPG.pt")
    # left_agents["b_PPG_I"] = PPGAgentWrapper("b_PPG_I", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/b_PPG_I.pt")
    # left_agents["b_DQN_I"] = DQNAgent("b_DQN_I", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/b_DQN_I.pt")
    # left_agents["b_PPO"] = PPGAgentWrapper("b_PPO", NatureCNN, (4,84,84), 6, "./league_models/b_PPO.pt")
    # Add 5th...
    left_agents["beta"] = PPGAgentWrapper("beta", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/beta.pt")

    # --- Loading Right Side ---
    # right_agents["a_PPG"] = PPGAgentWrapper("a_PPG", NatureCNN, (4,84,84), 6, "./league_models/a_PPG.pt")
    # right_agents["a_PPG_I"] = PPGAgentWrapper("a_PPG_I", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/a_PPG_I.pt")
    # right_agents["a_DQN_I"] = DQNAgent("a_DQN_I", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/a_DQN_I.pt")
    # right_agents["a_PPO"] = PPGAgentWrapper("a_PPO", NatureCNN, (4,84,84), 6, "./league_models/a_PPO.pt")
    # Add 5th...
    right_agents["alpha"] = PPGAgentWrapper("alpha", ImpalaCNN_DQN, (4,84,84), 6, "./league_models/alpha.pt")

    all_agents = {**left_agents, **right_agents}
    print(f">>> Loaded {len(all_agents)} Warriors.", flush=True)

    # 2. Matchmaking Setup
    l_keys = list(left_agents.keys())
    r_keys = list(right_agents.keys())
    
    prev_act = collections.defaultdict(lambda: None)
    prev_meta = collections.defaultdict(lambda: None)
    
    # Initial random pairings
    matchups = []
    for _ in range(NUM_ENVS):
        matchups.append((random.choice(l_keys), random.choice(r_keys)))

    # 3. Training Loop
    print(">>> Starting The Arena...", flush=True)
    for round_idx in range(1, TOTAL_ROUNDS+1):

        # --- TRAINING ROLLOUT ---
        for step in range(1, ROLLOUT_STEPS+1):
            if step % 100 == 0:
                print(f"   > Rollout Step {step}/{ROLLOUT_STEPS} (GPU Active)", flush=True)

            # A. Prepare Batch
            batch_obs = collections.defaultdict(list)
            batch_idxs = collections.defaultdict(list)
            
            for env_i in range(NUM_ENVS):
                pL, pR = matchups[env_i]
                
                # Left
                batch_obs[pL].append(obs[env_i*2])
                batch_idxs[pL].append(env_i*2)
                
                # Right
                batch_obs[pR].append(obs[env_i*2])
                batch_idxs[pR].append(env_i*2+1)
            
            # B. Inference
            global_actions = np.zeros(NUM_ENVS*2, dtype=int)
            for name, obs_list in batch_obs.items():
                if not obs_list: continue

                t_obs = torch.tensor(np.array(obs_list)).to(DEVICE)
                actions, meta = all_agents[name].get_action_batch(t_obs)
                
                idxs = batch_idxs[name]
                global_actions[idxs] = actions.cpu().numpy()
                
                for i, g_idx in enumerate(idxs):
                    prev_act[g_idx] = actions[i]

                    if meta is not None:
                        single_meta = {}
                        
                        for k, v in meta.items():
                            single_meta[k] = v[i]
                        prev_meta[g_idx] = single_meta
                    
                    else:
                        prev_meta[g_idx] = None
            
            # C. Step
            next_obs, rewards, term, trunc, _ = env.step(global_actions)
            dones = term | trunc
            
            # D. Store Experience
            for name, idxs in batch_idxs.items():
                s = torch.tensor(np.array([batch_obs[name][i] for i in range(len(idxs))])).to(DEVICE)
                
                ns_list = []
                for g_idx in idxs:
                    raw = next_obs[g_idx]
                    ns_list.append(raw)
                ns = torch.tensor(np.array(ns_list)).to(DEVICE)
                
                a = torch.stack([prev_act[i] for i in idxs]).to(DEVICE)
                r = torch.tensor(rewards[idxs], dtype=torch.float32).to(DEVICE)
                d = torch.tensor(dones[idxs], dtype=torch.uint8).to(DEVICE)
                
                # Meta handling
                m = {}
                if prev_meta[idxs[0]]:
                    for k in prev_meta[idxs[0]]:
                        m[k] = torch.stack([
                            prev_meta[i][k] if isinstance(prev_meta[i][k], torch.Tensor) 
                            else torch.tensor(prev_meta[i][k]).to(DEVICE) 
                            for i in idxs
                        ])
                
                all_agents[name].observe_batch(s, a, m, r, d, ns)

            obs = next_obs
        
        # --- UPDATES ---
        print(">>> Updating Agents...", flush=True)
        
        # 1. Map Agents to their specific Environment Indices
        # This handles dynamic matchmaking (e.g., if you have 10 agents playing randomly)
        agent_indices = collections.defaultdict(list)
        for env_i, (pL, pR) in enumerate(matchups):
            agent_indices[pL].append(env_i * 2)      # Left agent is at even index
            agent_indices[pR].append(env_i * 2 + 1)  # Right agent is at odd index

        logs = {}
        
        # 2. Iterate through all agents and update them
        for name, agent in all_agents.items():
            # Get the indices where this agent was playing
            idxs = agent_indices[name]
            
            # If the agent didn't play this round, skip update
            if not idxs: 
                continue 

            # 3. Extract Bootstrap Data (The "Step 129" state)
            # We grab the final 'obs' and 'dones' only for the indices this agent controlled.
            
            # Use list comprehension or np.take to gather specific non-contiguous rows
            batch_bootstrap_obs = np.array([obs[i] for i in idxs])
            batch_bootstrap_dones = np.array([dones[i] for i in idxs])
            
            # Convert to Tensor
            t_bootstrap_obs = torch.tensor(batch_bootstrap_obs).to(DEVICE)
            t_bootstrap_dones = torch.tensor(batch_bootstrap_dones, dtype=torch.uint8).to(DEVICE)
            
            # 4. Update with Bootstrapping
            loss = agent.update(
                next_obs_batch=t_bootstrap_obs, 
                next_done_batch=t_bootstrap_dones
            )
            
            if loss is not None:
                logs[f"train/{name}_loss"] = loss

        wandb.log(logs, step=round_idx)

        # --- SAVE ---
        if round_idx % SAVE_INTERVAL == 0:
            print(">>> Saving Checkpoints...", flush=True)
            for agent in all_agents.values():
                agent.save()

        # --- EVALUATION ---
        if round_idx > 0 and round_idx % EVAL_INTERVAL == 0:
            # run_evaluation_tournament(left_agents, right_agents, round_idx)
            print(">>> Resuming Training...", flush=True)
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
            print("", flush=True)

        # --- SHUFFLE MATCHUPS ---
        matchups = []
        for _ in range(NUM_ENVS):
            matchups.append((random.choice(l_keys), random.choice(r_keys)))
            
    wandb.finish()
    env.close()

if __name__ == "__main__":
    main()