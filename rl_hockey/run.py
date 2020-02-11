from PIL import Image, ImageTk
from rl_hockey.physics import rigid_body_physics
import torch
import time
import numpy as np


def run_and_train(params, memory, world, cpu_controller, canvas, root):
    loss_mean = 0
    if params['render']:  # If we are viewing the run pass the necessary arguments
        memory, num_frames = run_episode(memory, world=world, canvas=canvas, root=root, draw_step=2, numSteps=20_000)
    else:
        memory, num_frames = run_episode(memory, world=world, numSteps=1500)  # Run an iteration
        for j in range(len(cpu_controller)):  # For each CPU controller
            n_train_steps = 100
            for k in range(n_train_steps):  # Run a number of training steps
                loss_mean += cpu_controller[j].train(memory[j], params['beta'])
            loss_mean = loss_mean / (n_train_steps)

    return num_frames, loss_mean


def run_episode(memory, world, numSteps=1500, canvas=None, root=None, draw_step=1, draw_scale=0.5):
    """
    This function runs through an entire episode. Each frame will be stored in the
    memory object for future traning.
    """

    # Initialize world
    world.reset()
    num_cpu = world.get_num_cpu()
    frame_skip = 4
    timestep = 1 / 30

    # Initial state and score
    state = world.get_state()
    score = world.get_scores()

    i = 0
    while i <= numSteps:
        # Update memory ever n=frame_skip frames
        if i % frame_skip == 0 and i > 0:
            prev_score = score
            score = world.get_scores()
            done = world.terminate_run()
            prev_state = state.copy()
            state = world.get_state()
            all_actions = world.get_last_action()
            impacts = world.get_impact()
            for j in range(num_cpu):
                reward = torch.tensor([(score[j] - prev_score[j])], dtype=torch.float)
                action = torch.tensor([all_actions[j]])
                memory[j].push(prev_state[j], action, reward, state[j], done=done, impact=impacts[j])

            if done:  # Check if we should terminate, then do so
                break

        # On first frame of block apply control
        if i % frame_skip == 0:
            world.apply_control(state=state, frame_skip=False)  # Update state of world
        else:
            world.apply_control(state=state, frame_skip=True)  # Update state of world

        rigid_body_physics(world, timestep)        # Apply physics
        world.update_score()                       # Update world, including score

        # Render world if canvas is provided
        if canvas is not None and i % draw_step == 0:
            render_world(canvas, root, world, draw_scale)

        i += 1

    return memory, i

def render_world(canvas, root, world, draw_scale=0.5):
    """
    Render world onto given canvas
    """
    arr = world.draw_world(high_scale=draw_scale)
    arr = arr.transpose([1, 0, 2])

    img = ImageTk.PhotoImage(image=Image.fromarray(arr.astype(np.uint8), mode='RGB'))
    canvas.create_image(0, 0, anchor='nw', image=img)
    root.update()

def post_episode(params, memory, world, cpu_controller, self_play_cpu, cur_episode, score_hist, results, total_frames, num_frames, time_elapsed):
    if world.check_self_play():  # If self play is used, increment the internal counters
        for j in range(len(self_play_cpu)):  # of the self play controllers.
            self_play_cpu[j].increment_model()
            if cur_episode % 1000 == 0:  # Every 1000 iterations take a snapshot of controllers
                self_play_cpu[j].insert_model(cpu_controller[j].get_model(), cpu_controller[j].get_eps())

    if cur_episode % 100 == 0 and params['render'] is False:  # Every 100 iterations save the current model(s)
        if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):
            cpu_controller[0].save_model('./trained_models/' + params['save_name'] + '.pt')

    # Get current score and provide some output
    score_hist.append(np.mean(world.get_scores()[:world.get_num_cpu()]))
    score_mean = np.mean(score_hist[np.max([cur_episode - 100, 0]):])
    results.append([total_frames, num_frames, score_hist[-1]])

    print('Iteration {}, memLen {}, frames {:.6f}, time {:.2f}, score {}, '
          'avg_score {:.2f}'.format(cur_episode, len(memory[0]), total_frames, time_elapsed,
                                    world.get_scores()[:world.get_num_cpu()], score_mean))

