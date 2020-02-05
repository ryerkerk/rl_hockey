from rl_hockey.world import *
from rl_hockey.controller import (DoubleDQN, DQN, Buffer, NaivePrioritizedBuffer, SelfPlayController, ImpactPrioritizedMemory)
import os
import time
import tkinter as tk
import numpy as np
from rl_hockey.run import run
from rl_hockey.utils import parse_arg
import pickle

#BETA_START = 0.4                # Initial beta value used by prioritized memory
#BETA_MAX = 0.5                  # Final beta value
#BETA_FRAMES = 1000              # Transition period from start to max veta value
#beta_by_frame = lambda frame_idx: min(BETA_MAX, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

if __name__ == "__main__":
    params = parse_arg()
    print(params)

    # Set up world and required controllers.
    if params['algorithm'] == 'dqn':
        print('Using dqn trainer')
        controller = DQN
    elif params['algorithm'] == 'double_dqn':
        print('Using double_dqn dqn trainer')
        controller = DoubleDQN
    else:
        raise Exception('Controller type not recognized')

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

    if params['world'] == 'Hockey1v1':
        SAVE_NAME = 'Hockey1v1_prior'
        world = Hockey1v1()                                     # Create world
        SELF_PLAY = True                                        # This world using self play
        memory = [memory_type(params['mem_capacity'])]             # Initialize empty memory
        num_actions = world.get_num_actions()                   # Number of actions available for agent
        num_inputs = len(world.get_state()[0][0])               # Number of inputs provided
        cpu_controller = [controller(device=params['device'], network_type=params['network'], optimizer_type=params['optimizer'])] # Create controller
        cpu_controller[0].create_model(num_inputs=num_inputs, num_actions=num_actions[0], gamma=params['gamma'], eps_end=params['eps_end'],
                                       eps_decay=params['eps_decay'], lr=params['lr'], hn=params['hn_size'])

    elif params['world'] == 'Hockey1v1Heuristic':
        SAVE_NAME = 'Hockey1v1Heuristic'
        world = Hockey1v1Heuristic()                            # Create world
        SELF_PLAY = False                                       # World using an heuristic opponent
        memory = [memory_type(params['mem_capacity'], age_buffer=0.5)]             # Initialize empty memory
        num_actions = world.get_num_actions()                   # Number of actions available for agent
        num_inputs = len(world.get_state()[0][0])               # Number of inputs provided
        cpu_controller = [controller(device=params['device'], network_type=params['network'], optimizer_type=params['optimizer'])]  # Create controller
        cpu_controller[0].create_model(num_inputs=num_inputs, num_actions=num_actions[0], gamma=params['gamma'], eps_end=params['eps_end'],
                                       eps_decay=params['eps_decay'], lr=params['lr'], hn=params['hn_size'])
    else:
        raise Exception('WORLD not recognized')

    # Currently if a PREVIOUS_MODEL is provided then it is not intended to be trained, but rather viewed.
    if params['initial_model'] is not 'none':
        fname = "./trained_models/{}.pt".format(os.path.splitext(params['initial_model'])[0])      # Strip extension if necessary
        if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):  # If all CPU controllers are same
            cpu_controller[0].load_model(fname)  # Load model into controller
            cpu_controller[0].train_steps = params['eps_decay'] * 100       # Set train steps high so a low eps is used

        # params['render'] = True         # Modify in future, currently can't properly load then train. Need to save memory

    if params['render'] is True:        # If a viewer is going to be used then setup the tk.Canvas
        world_size = world.get_world_size()
        print(world_size)
        root = tk.Tk()
        canvas = tk.Canvas(root, width=world_size[0] * params['draw_scale'], height=world_size[1] * params['draw_scale'])
        canvas.pack()

    # If self play is being used for training then we set up a SelfPlayController.
    # Occasional snapshots of the trained controller will be stored here to be used as an opponent
    if SELF_PLAY:                       # If we're doing self play create controller and load initial model
        self_play_cpu = []              # Create a self play controller for each CPU controller
        for c in cpu_controller:
            self_play_cpu.append(SelfPlayController(num_actions=num_actions))
            self_play_cpu[-1].insert_model(c.get_model(), c.get_eps())         # Load initial controller
    else:  # No self play, world will rely on it's own heuristic controllers
        self_play_cpu = None

    world.set_cpu_controller(cpu_controller, self_play_cpu)     # Insert controllers to world

    start = time.time()
    score_hist = []

    total_frames = 0
    cur_episode = 0
    results = []
    while cur_episode <= params['total_episodes']:
        start = time.time()
        loss_mean = 0
        if params['render']:                    # If we are viewing the run pass the necessary arguments
            memory = run(memory, world=world, canvas=canvas, root=root, draw_step=2, pause_time=1/45, numSteps=20_000)
        else:
            memory, num_frames = run(memory, world=world, numSteps=1500)    # Run an iteration
            total_frames += num_frames
            for j in range(len(cpu_controller)):                # For each CPU controller
                n_train_steps = 100
                for k in range(n_train_steps):          # Run a number of training steps
                    loss_mean += cpu_controller[j].train(memory[j], params['beta'])
                loss_mean = loss_mean / ( n_train_steps )

        stop = time.time()

        cur_episode += 1

        if SELF_PLAY is True:                                   # If self play is used, increment the internal counters
            for j in range(len(self_play_cpu)):                 # of the self play controllers.
                self_play_cpu[j].increment_model()
                if cur_episode % 1000 == 0:                               # Every 1000 iterations take a snapshot of controllers
                    self_play_cpu[j].insert_model(cpu_controller[j].get_model(), cpu_controller[j].get_eps())

        if cur_episode % 100 == 0 and params['render'] is False:                  # Every 100 iterations save the current model(s)
            if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):
                cpu_controller[0].save_model('./trained_models/' + params['save_name'] + '.pt')

        # Get current score and provide some output
        score_hist.append(np.mean(world.get_scores()[:world.get_num_cpu()]))
        score_mean = np.mean(score_hist[np.max([cur_episode-100, 0]):])
        results.append([total_frames, num_frames, score_hist[-1]])

        print('Iteration {}, memLen {}, frames {:.6f}, time {:.2f}, score {}, avg_score {:.2f}'.format(cur_episode, len(memory[0]),
              total_frames, stop-start, world.get_scores()[:world.get_num_cpu()], score_mean))

    pickle.dump(results, open('./results/' + params['save_name'] + '.p', 'wb'))