import random

import torch
import torch.optim as optim
from .controller import Controller
from .network import FFNetwork, DuelingFFNetwork


class DQN(Controller):
    """
       This controller is a DQN. It will be extended to a double DQN below. A few class properties are included in this
       class that are specific to the double DQN (i.e. target_net), and are unused here.

       This class is adapted from the following:
       The repository by Dulat Yerzat: https://github.com/higgsfield/RL-Adventure
       The DQN tutorial by Adam Paszke (https://github.com/apaszke)
       """

    def __init__(self, device='cuda', network_type='feed_forward', optimizer_type='adagrad'):
        super(DQN, self).__init__()
        self.device = torch.device(device)
        self.train_steps = 0

        # These will get set when creating model
        self.num_actions = 0
        self.gamma = 0.99
        self.eps_end = 0.1
        self.eps_decay = 50_000
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.network_type = network_type
        self.optimizer_type = optimizer_type

    def create_model(self, num_inputs, num_actions, gamma=0.99, eps_end=0.1, eps_decay=50_000, lr=1e-2, hn=512):
        """
        Initialize the model for the given parameters.

        :param num_inputs: Number of state values used as inputs
        :param num_actions: Number of potential actions
        :param gamma: Gamma value used in DQN
        :param eps_end: Final eps value
        :param eps_decay: Rate at which eps will decay
        :param lr: Learning rate
        :param hn: Number of hidden nodes in each layer of DQN
        :return:
        """
        self.num_actions = num_actions
        self.train_steps = 0  # Initialize number of training steps to 0
        self.gamma = gamma  # Assign invalue values
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # Create policy and target nets, then assign initial weights from policy net to target net
        if self.network_type == 'feed_forward':
            print('Using a feed forward network')
            self.policy_net = FFNetwork(num_inputs=num_inputs, num_outputs=num_actions, hn=hn).to(self.device)
            self.target_net = FFNetwork(num_inputs=num_inputs, num_outputs=num_actions, hn=hn).to(self.device)
        elif self.network_type == 'dueling_feed_forward':
            print('Using a dueling feed forward network')
            self.policy_net = DuelingFFNetwork(num_inputs=num_inputs, num_outputs=num_actions, hn=hn).to(self.device)
            self.target_net = DuelingFFNetwork(num_inputs=num_inputs, num_outputs=num_actions, hn=hn).to(self.device)
        else:
            raise Exception('Unrecognized network type')

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer to be used
        if self.optimizer_type == 'adagrad':
            self.optimizer = optim.Adagrad(self.policy_net.parameters(), lr=lr)
        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr)
        else:
            raise Exception('Unrecognized optimizer')
        print(self.policy_net.forward)
        pass

    def select_action(self, state):
        """
        The selected action will either be determined by the policy net, or randomly determined.

        The rate at which the action is randomly chosen is based on the current eps value. When few training steps
        have been performed most actions will be randomly chosen. Over time the policy net will be used more frequently
        to determine actions. The final rate of randomly chosen actions is determined by self.eps_end
        """

        state = torch.tensor(state, dtype=torch.float32)
        sample = random.random()
        eps = self.get_eps()  # Get current eps value

        if sample > eps:  # Determine action using policy net
            with torch.no_grad():
                self.policy_net.eval()
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state.to(self.device)).max(1)[1]
        else:  # Determine action randomly
            return torch.tensor([random.randrange(self.num_actions)], device=self.device, dtype=torch.long)

    def train(self, memory, beta):
        """
        Training us performed using DQN. This function is primarily adapted from the repositories listed
        at the start of this class.
        """
        BATCH_SIZE = 128
        GAMMA = self.gamma

        if len(memory) < BATCH_SIZE:
            return 0

        if len(memory) < 10000:  # Don't start training off of very initial memories
            return 0

        # Sample memory
        state, action, reward, next_state, done, indices, weights = memory.sample(BATCH_SIZE, beta)

        # Get memories into proper forms
        state = [torch.tensor(b, dtype=torch.float32) for b in state]
        next_state = [torch.tensor(b, dtype=torch.float32) for b in next_state]
        done = torch.tensor(done, dtype=torch.uint8)
        state_batch = torch.cat(state).to(self.device)
        next_batch = torch.cat(next_state).to(self.device)
        action_batch = torch.cat(action).reshape(-1, 1).to(self.device)
        reward_batch = torch.cat(reward).reshape(-1, 1).to(self.device)
        weights = torch.tensor(weights).reshape(-1, 1).to(self.device)

        self.policy_net.eval()

        # Double DQN
        self.policy_net.train()

        Q = self.policy_net(state_batch).gather(1, action_batch)
        next_Q = self.policy_net(next_batch).max(1)[0].detach()
        next_Q[done == 1] = 0
        next_Q = next_Q.view(-1, 1)
        expected_Q = next_Q*GAMMA + reward_batch

        # Compute loss

        diff = Q  - expected_Q
        loss = (0.5 * (diff * diff)) * weights

        # Update memory priorities
        priors = loss + 1e-5
        memory.update_priorities(indices, priors.data.cpu().numpy())  # Update

        # Optimize the model
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():  # Confirm this
            param.grad.data.clamp(-1, 1)

        self.optimizer.step()

        self.train_steps += 1
        unweighted_loss = (0.5 * (diff * diff)).mean()  # Unweighted loss used only for output
        return unweighted_loss.detach().cpu().numpy()

    def save_model(self, path):
        """
        Save the current policy net model to the given path
        """
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """
        Load the policy net model from the given path
        """
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_model(self):
        return self.policy_net


class DoubleDQN(DQN):
    """
    This controller is based on a double DQN model. It can use the same methods as the DQN controller except for train()

    This class is adapted from the following:
    The repository by Dulat Yerzat: https://github.com/higgsfield/RL-Adventure
    The DQN tutorial by Adam Paszke (https://github.com/apaszke)
    Double DQN was proposed by van Hasselt et al. "Deep Reinforcement Learning with Double Q-learning" (2015) https://arxiv.org/abs/1509.06461
    """

    def train(self, memory, beta):
        """
        Training us performed using double DQN. This function is primarily adapted from the repositories listed
        at the start of this class.
        """
        BATCH_SIZE = 128
        GAMMA = self.gamma

        if len(memory) < BATCH_SIZE:
            return 0

        if len(memory) < 10000:  # Don't start training off of very initial memories
            return 0

        # Sample memory
        state, action, reward, next_state, done, indices, weights = memory.sample(BATCH_SIZE, beta)

        # Get memories into proper forms
        state = [torch.tensor(b, dtype=torch.float32) for b in state]
        next_state = [torch.tensor(b, dtype=torch.float32) for b in next_state]
        done = torch.tensor(done, dtype=torch.uint8)
        next_batch = torch.cat(next_state).to(self.device)
        state_batch = torch.cat(state).to(self.device)
        action_batch = torch.cat(action).reshape(-1, 1).to(self.device)
        reward_batch = torch.cat(reward).reshape(-1, 1).to(self.device)
        weights = torch.tensor(weights).reshape(-1, 1).to(self.device)

        self.policy_net.eval()
        self.target_net.eval()

        # Double DQN
        with torch.no_grad():
            online_Q = self.policy_net(next_batch)
            target_Q = self.target_net(next_batch)
            next_Q = target_Q.gather(1, online_Q.max(1)[1].detach().unsqueeze(1)).squeeze(1)
            next_Q[done == 1] = 0
            target_Q = next_Q * GAMMA + reward_batch.flatten()

        # Compute loss
        self.policy_net.train()
        current_Q = self.policy_net(state_batch).gather(1, action_batch)
        diff = current_Q.squeeze() - target_Q
        loss = (0.5 * (diff * diff))*weights.squeeze()

        # Update memory priorities
        prios = loss + 1e-5
        memory.update_priorities(indices, prios.data.cpu().numpy()) # Update

        # Optimize the model
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():  # Confirm this
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

        # Every 1000 training steps copy set the target net to the current policy net
        if self.train_steps % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.train_steps += 1
        unweighted_loss = (0.5 * (diff * diff)).mean() # Unweighted loss used only for output
        return unweighted_loss.detach().cpu().numpy()



