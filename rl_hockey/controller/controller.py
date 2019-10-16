import math


class Controller():
    """
    Parent class for controllers. No functionality, was meant to provide a template for controllers to follow

    """
    def __init__(self):
        pass

    def create_model(self, **kwargs):
        pass

    def select_action(self, state):
        pass

    def get_input(self):
        pass

    def train(self, memory, beta):
        pass

    def save_model(self, PATH):
        pass

    def load_model(self, PATH):
        pass

    def get_eps(self):
        """
        Calculate current eps based on number of elapsed training steps.
        """
        eps_start = 0.9
        eps_end = self.eps_end
        eps_decay = self.eps_decay

        return eps_end + (eps_start - eps_end) * math.exp(-1. * self.train_steps / eps_decay)