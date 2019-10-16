import torch.nn as nn

class FFNetwork(nn.Module):
    """
    This is a simple feed forward network. The number of inputs will be equal to the size of the provided world state,
    and the number of outputs equal to the number of potential actions.
    """
    def __init__(self, num_inputs, num_outputs, hn):
        super(FFNetwork, self).__init__()

        self.num_actions = num_outputs
        num_hidden = hn
        nh2 = int(num_hidden / 2)

        self.features = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, nh2),
            nn.ReLU(),
            nn.Linear(nh2, num_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class DuelingFFNetwork(nn.Module):
    """
    The dueling architecture considers both the value in taking particular action, and the value of being
    at the current state.

    This class is adapted from the following repository: https://github.com/higgsfield/RL-Adventure
    and was based on Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2015)
    https://arxiv.org/abs/1511.05952
    """
    def __init__(self, num_inputs, num_outputs, hn):
        super(DuelingFFNetwork, self).__init__()

        self.num_actions = num_outputs
        num_hidden = hn
        nh2 = int(num_hidden/2)

        self.features = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(num_hidden, nh2),
            nn.ReLU(),
            nn.Linear(nh2, num_outputs),
        )

        self.value = nn.Sequential(
            nn.Linear(num_hidden, nh2),
            nn.ReLU(),
            nn.Linear(nh2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        out = value + advantage - advantage.mean()
        return out