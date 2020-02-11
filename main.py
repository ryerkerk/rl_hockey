from rl_hockey.world import *
from rl_hockey.controller import (DoubleDQN, DQN, Buffer, NaivePrioritizedBuffer, SelfPlayController, ImpactPrioritizedMemory)
import os
import time
import tkinter as tk
import numpy as np
from rl_hockey.run import run_episode, run_and_train, post_episode
from rl_hockey.utils import parse_arg, setup_world_controller_memory, load_previous_model, create_canvas
import pickle

if __name__ == "__main__":
    params = parse_arg()
    print(params)

    world, cpu_controller, memory = setup_world_controller_memory(params)

    # Load previous model, if necessary
    load_previous_model(params, cpu_controller)

    if params['render'] is True:        # If a viewer is going to be used then setup the tk.Canvas
        canvas, root = create_canvas(params, world)
    else:
        canvas, root = None, None

    # If self play is being used for training then we set up a SelfPlayController.
    # Occasional snapshots of the trained controller will be stored here to be used as an opponent
    if world.check_self_play():         # If we're doing self play create controller and load initial model
        self_play_cpu = []              # Create a self play controller for each CPU controller
        for c in cpu_controller:
            self_play_cpu.append(SelfPlayController(num_actions=world.get_num_actions()))
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

        num_frames, loss_mean = run_and_train(params, memory, world, cpu_controller, canvas, root)
        total_frames += num_frames
        stop = time.time()

        cur_episode += 1
        post_episode(params, memory, world, cpu_controller, self_play_cpu, cur_episode, score_hist, results,
                     total_frames, num_frames, time_elapsed=stop-start)

    pickle.dump(results, open('./results/' + params['save_name'] + '.p', 'wb'))

