import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BLOCKS FOR IMPALA ---
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

# --- MAIN NETWORKS ---

class ImpalaCNN(nn.Module):
    # Modified signature to handle 'input_shape' if passed as tuple, or 'input_channels' if passed as int
    def __init__(self, input_channels=4, depths=[16, 32, 32], output_shape=6):
        super().__init__()
        
        # Handle if input_channels is passed as a tuple (C, H, W) by accident
        if isinstance(input_channels, (tuple, list)):
            input_channels = input_channels[0]
            
        # 1. DEFINE FEATURES
        self.features = nn.Sequential(
            ImpalaBlock(input_channels, depths[0]),
            ImpalaBlock(depths[0], depths[1]),
            ImpalaBlock(depths[1], depths[2]),
            nn.ReLU()
        )
        
        # 2. CALCULATE SIZE
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            dummy_output = self.features(dummy_input)
            self.linear_input_size = dummy_output.view(1, -1).shape[1]

        # 3. DEFINE HEAD
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
    

class NatureCNN(nn.Module):
    # CHANGED: Now accepts 'input_channels' (int) and 'depths' (ignored) to match Impala signature
    def __init__(self, input_channels=4, output_shape=6, depths=None):
        super().__init__()
        
        # Handle if input_channels is passed as a tuple (C, H, W)
        if isinstance(input_channels, (tuple, list)):
            n_input_channels = input_channels[0]
        else:
            n_input_channels = input_channels
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamic size calculation
        with torch.no_grad():
            # We assume 84x84 input if only channels are given
            dummy_input = torch.zeros(1, n_input_channels, 84, 84)
            n_flatten = self.cnn(dummy_input).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flatten, 512), 
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.head(x)
        return x