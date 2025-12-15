import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class CaptureFeatures:
    """Hooks into a layer to capture its input (features) during forward pass."""
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        # input is a tuple (x,), we want x[0]
        self.features = input[0]
    
    def close(self):
        self.hook.remove()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + inputs

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    def __init__(self, input_channels=4, depths=[16, 32, 32], output_shape=6):
        super().__init__()
        
        # 1. DEFINE FEATURES (The Convolutional Part)
        self.features = nn.Sequential(
            ImpalaBlock(input_channels, depths[0]),
            ImpalaBlock(depths[0], depths[1]),
            ImpalaBlock(depths[1], depths[2]),
            nn.ReLU()
        )
        
        # 2. CALCULATE SIZE (The "Dummy" Part)
        # We simulate one pass to see how big the flat vector will be.
        with torch.no_grad():
            # Create a fake image: Batch size 1, input_channels, 84x84 image
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            
            # Pass it through features
            dummy_output = self.features(dummy_input)
            
            # Calculate total parameters (e.g., 32 * 11 * 11 = 3872)
            self.linear_input_size = dummy_output.view(1, -1).shape[1]

        # 3. DEFINE HEAD (The Linear Part)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
    
class PPG_I(nn.Module):
    def __init__(self, action_dim=6, input_channels=4, device="cpu"):
        super().__init__()
        self.device = device
        
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
        # We assume the last linear layer in the actor is the output head.
        modules = list(self.actor.modules())
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        
        if len(linear_layers) > 0:
            self.actor_feature_hook = CaptureFeatures(linear_layers[-1])
            feature_dim = linear_layers[-1].in_features
        else:
            raise ValueError("Could not find a Linear layer in ImpalaCNN to hook features from.")

        # 4. Auxiliary Value Head
        self.aux_critic = nn.Linear(feature_dim, 1)

        # Init aux head weights (optional during inference, but good for consistency)
        torch.nn.init.orthogonal_(self.aux_critic.weight, 1.0)
        torch.nn.init.constant_(self.aux_critic.bias, 0.0)

        self.to(self.device)

    def get_value(self, x):
        out = self.critic(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def get_action_and_value(self, x, action=None):
        # Run Actor (Hook captures features automatically)
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
    
    # Añadir esto dentro de la clase PPG_I
    def predict(self, obs, deterministic=False):
        """
        Método compatible con Stable Baselines 3 para inferencia.
        """
        # Asegurarse de que obs es un Tensor y tiene dimensión de batch
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        # Si la observación viene sin dimensión de batch (C, H, W), añadirla -> (1, C, H, W)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        # Si viene como (H, W, C) que es común en Gym, puede que necesites permutar a (1, C, H, W)
        # Dependerá de tus wrappers. Asumiendo que ya llega correcto o el wrapper lo maneja:

        with torch.no_grad():
            logits = self.actor(obs)
            if isinstance(logits, tuple): logits = logits[0]
            
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
        
        # Devolver acción (como numpy int) y estado (None para este modelo)
        return action.cpu().numpy()[0], None
    
# Path to the agent's models
# Ensure these paths are correct relative to where you run the script
current_dir = os.path.dirname(os.path.abspath(__file__))

agent_left_name = os.path.join(current_dir, "beta.pt")
agent_right_name = os.path.join(current_dir, "alpha.pt")

def load_agent(model_path):
    """Helper function to load a single agent."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. Initialize the Architecture
    # Pong typically has 6 actions and 4 stacked frames (input_channels)
    agent = PPG_I(action_dim=6, input_channels=4, device="cpu")
    
    # 2. Load the Weights
    # map_location='cpu' ensures it loads even if trained on Cuda
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if checkpoint is the full state dict or just the model weights
    # (Your training script saves `agent.state_dict()`, so we load it directly)
    agent.load_state_dict(checkpoint, strict=False)
    
    # 3. Set to Evaluation Mode
    agent.eval()
    
    return agent

def load_right_agent():
    '''
    Loads and returns the right agent model.
    '''
    print(f"Loading Right Agent from {agent_right_name}...")
    agent = load_agent(agent_right_name)
    return agent

def load_left_agent():
    '''
    Loads and returns the left agent model.
    '''
    print(f"Loading Left Agent from {agent_left_name}...")
    agent = load_agent(agent_left_name)
    return agent