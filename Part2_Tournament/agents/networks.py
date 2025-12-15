import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class NatureCNN(nn.Module):
    def __init__(self, input_shape, output_shape=6):
        super().__init__()
        
        n_input_channels = input_shape[0]
        
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
            dummy_input = torch.zeros(1, *input_shape)
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