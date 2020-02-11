from rl_hockey.world import *
from rl_hockey.controller import (DoubleDQN, DQN, Buffer, NaivePrioritizedBuffer, SelfPlayController, ImpactPrioritizedMemory)
import os
import tkinter as tk

def setup_world_controller_memory(params):
    """
    Use the provided parameters to initialize the world, controller, and memory
    """

    algorithm_type, memory_type = get_algorithm_memory_type(params)

    # Setup world
    if params['world'] == 'Hockey1v1':
        world = Hockey1v1()  # Create world
    elif params['world'] == 'Hockey1v1Heuristic':
        world = Hockey1v1Heuristic()  # Create world
    else:
        raise Exception('WORLD not recognized')

    # Setup memory and controller
    memory = [memory_type(params['mem_capacity'], age_buffer=0.5)]  # Initialize empty memory
    num_actions = world.get_num_actions()  # Number of actions available for agent
    num_inputs = len(world.get_state()[0][0])  # Number of inputs provided
    cpu_controller = [algorithm_type(device=params['device'], network_type=params['network'],
                                     optimizer_type=params['optimizer'])]  # Create controller
    cpu_controller[0].create_model(num_inputs=num_inputs, num_actions=num_actions[0], gamma=params['gamma'],
                                   eps_end=params['eps_end'],
                                   eps_decay=params['eps_decay'], lr=params['lr'], hn=params['hn_size'])

    return world, cpu_controller, memory


def get_algorithm_memory_type(params):
    """
    Check the given parameters and return handles to the desired algorithm and memory
    type to be used by controller.
    """
    # Get handle of algorithm to be used
    if params['algorithm'] == 'dqn':
        print('Using dqn trainer')
        algorithm_type = DQN
    elif params['algorithm'] == 'double_dqn':
        print('Using double_dqn dqn trainer')
        algorithm_type = DoubleDQN
    else:
        raise Exception('Controller type not recognized')

    # Get handle to desired memory object
    if params['memory'] == 'prioritized':
        print('Using prioritized memory')
        memory_type = NaivePrioritizedBuffer
    elif params['memory'] == 'impact_prioritized':
        print('Using impact + prioritized memory')
        memory_type = ImpactPrioritizedMemory
    elif params['memory'] == 'buffer':
        print('Using basic buffer memory')
        memory_type = Buffer
    else:
        raise Exception('Memory type not recognized')

    return algorithm_type, memory_type


def load_previous_model(params, cpu_controller):
    # Currently if a PREVIOUS_MODEL is provided then it is not intended to be trained, but rather viewed.
    if params['initial_model'] is not 'none':
        fname = "./trained_models/{}.pt".format(
            os.path.splitext(params['initial_model'])[0])  # Strip extension if necessary
        if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):  # If all CPU controllers are same
            cpu_controller[0].load_model(fname)  # Load model into controller
            cpu_controller[0].train_steps = params['eps_decay'] * 100  # Set train steps high so a low eps is used

def create_canvas(params, world):
    world_size = world.get_world_size()
    root = tk.Tk()
    canvas = tk.Canvas(root, width=world_size[0] * params['draw_scale'], height=world_size[1] * params['draw_scale'])
    canvas.pack()
    return canvas, root