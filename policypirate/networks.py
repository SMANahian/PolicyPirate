"""Neural networks for the PPO agent (plugged into Stable-Baselines3)."""

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CnnDense(BaseFeaturesExtractor):
    """CNN + dense: three 3x3 convolutions over the board, then one dense layer of 256.

    SB3 puts its own final layers on top of these 256 features: one gives the 100 cell
    scores (the policy), the other the value estimate.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        layers, height, width = observation_space.shape
        self.net = nn.Sequential(
            nn.Conv2d(layers, 32, kernel_size=3, padding=1),  # padding=1 keeps the board 10x10
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


class ConvNet(BaseFeaturesExtractor):
    """Fully convolutional: six 3x3 convolutions that keep the 10x10 board, then a 1x1
    convolution that gives one score per cell.

    Output: the 100 cell scores, followed by the 64 conv features averaged over the board
    (used for the value estimate).
    """

    def __init__(self, observation_space, channels=64, depth=6):
        layers, height, width = observation_space.shape
        super().__init__(observation_space, features_dim=height * width + channels)
        convs = []
        for i in range(depth):
            convs += [nn.Conv2d(layers if i == 0 else channels, channels, kernel_size=3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*convs)
        self.cell_scores = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, obs):
        x = self.body(obs)  # (batch, 64, 10, 10)
        scores = self.cell_scores(x).flatten(1)  # (batch, 100)
        averaged = x.mean(dim=(2, 3))  # (batch, 64)
        return torch.cat([scores, averaged], dim=1)


class SplitHeads(nn.Module):
    """Sends the cell scores to the policy and the averaged features to the value head."""

    def __init__(self, num_cells, value_dim):
        super().__init__()
        self.num_cells = num_cells
        self.latent_dim_pi = num_cells  # SB3 reads these two sizes
        self.latent_dim_vf = value_dim

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return features[:, : self.num_cells]

    def forward_critic(self, features):
        return features[:, self.num_cells :]


class ConvPolicy(MaskableActorCriticPolicy):
    """SB3's masked actor-critic, with the cell scores taken straight from ConvNet."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, features_extractor_class=ConvNet, **kwargs)

    def _build_mlp_extractor(self):
        num_cells = self.action_space.n
        self.mlp_extractor = SplitHeads(num_cells, self.features_dim - num_cells)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()  # no dense layer on top of the cell scores
        # SB3 starts its last policy layer with small weights (gain 0.01) so the first moves are
        # nearly uniform; do the same for our last policy layer.
        self.init_weights(self.features_extractor.cell_scores, gain=0.01)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
