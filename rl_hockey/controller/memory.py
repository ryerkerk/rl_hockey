import numpy as np


class Memory:
    def __init__(self):
        pass

    def push(self, state, action, reward, next_state, done=0, **kwargs):
        pass

    def sample(self, batch_size, beta=0.4):
        pass

    def update_priorities(self, batch_indices, batch_priorities):
        pass

    def __len__(self):
        pass

class Buffer(Memory):
    """
    A basic memory buffer.

    This class is adapted from the following:
    The repository by Dulat Yerzat: https://github.com/higgsfield/RL-Adventure
    The DQN tutorial by Adam Paszke (https://github.com/apaszke)
    """
    def __init__(self, capacity, **kwargs):
        super(Buffer, self).__init__()

        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done=0, **kwargs):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        indices = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[idx] for idx in indices]
        weights = np.ones(batch_size, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(Memory):
    """
    The prioritized replay buffer assigns priorities to each memory based on their observed loss from the previous time
    sampled. Memories with higher priorities are more likely to get sampled, and are assigned higher weights when
    training the network.

    This class is taken from the following repository by Dulat Yerzat: https://github.com/higgsfield/RL-Adventure
    which was based on Schaul et al. "Prioritized Replay Memory" (2015) https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, prob_alpha=0.6, **kwargs):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.impact = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done=0, impact=1):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.impact[self.pos] = impact
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


class ImpactPrioritizedMemory(Memory):
    """
    This is an extension of NaivePrioritizedBuffer.

    In NaivePrioritizedBuffer memories are based entirely on age. Once the buffer is full new memories will replace the
    oldest remaining memories. In this class, each memory is assigned in "impact", this is a measure of the potential
    impact the agent might have had on the world at this state. For example, in the hockey worlds impact is measured
    as the distance of the agent to the puck. Agents very far from the puck have little impact.

    Once the buffer is full all memories are scored based on their age and impact. Memories with the worst score, those
    with relatively high ages and low impacts, will be marked to be replaced by new memories.
    """
    def __init__(self, capacity, prob_alpha=0.6, age_buffer=0.5, **kwargs):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = []       # Will contain index of memories to remove
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.impact = np.zeros((capacity,), dtype=np.float32)
        self.age = np.zeros((capacity,), dtype=np.float32)
        self.age_buffer = age_buffer;

    def push(self, state, action, reward, next_state, done=0, impact=1):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities[len(self.buffer)-1] = max_prio
            self.impact[len(self.buffer)-1] = impact
            self.age[len(self.buffer) - 1] = 0
        else:
            if len(self.pos) == 0:
                self.reset_pos()
            idx = self.pos.pop(0)
            self.buffer[idx] = (state, action, reward, next_state, done)
            self.age[idx] = 0
            self.priorities[idx] = max_prio
            self.impact[idx] = impact

        self.age = self.age + 1

    def reset_pos(self):
        """
        self.pos will contain the indices of the next memories to be replaced by new memories.

        Note that this function should only be called once the buffer is full.
        """

        score = self.impact / np.mean(self.impact) - self.age/int(2*self.capacity)
        score[self.age < int(self.capacity*self.age_buffer)] = np.max(score)
        self.pos = list(np.argsort(score)[:int(self.capacity*0.1)])


    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
