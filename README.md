![](https://raw.githubusercontent.com/ryerkerk/rl_hockey/master/sample_view.gif)

# rl_hockey
An exploration of applying reinforcement learning to an air hockey style game.

This project includes a simple rigid body physics engine, on top of which several variations of the game, or worlds, are built. These act as test beds for several reinforcement learning models and algorithms.

A number of parameters for the network architecture and training algorithm can be set as arguments. DQN and double DQN algorithms, feed forward and dueling feed forward network architectures, and basic, prioritized, or impact prioritized memories are supported.

Impact priortized memory replaces old memories based on age and a measure of impact. At the moment impact is measured as the distance of the agent to the puck. This can reduce the required learning time, but may also reduce model stability.

## Getting started

# Requirements

- Python (3.6)
- numpy
- Pytorch
- pillow
- pyglet

The provided `environment.yml` can be used to create an appropriate conda environment. A Dockerfile is also provided.

  `conda env update --file environment.yml --name rl_hockey`

# Worlds

Two variations of the hockey game, or worlds, are available here.   

- `Hockey1v1`: This is a 1v1 game where the controller is trained through self play.
- `Hockey1v1Heuristic`: The controller is trained by competing against a very basic bot.

In `Hockey1v1Heuristic` the agent plays against a heuristically coded opponent. The opponent chases the puck and then retreats toward the goal if the puck gets behind him.

In `Hockey1v1` self play is used for raining. Each iteration the controller will compete against a previous screenshot of itself. Every 1000 iterations a screenshot of the current controller(s) is taken, with the 5 most recently screenshots kept in memory.

In both worlds the current model will be saved to the filename specified by --save_name=<save_name> every 100 iterations. Restarting training is currently not supported.

# Training

The model can be trained by specifying a save name and world in which to train. Additional parameters are detailed in the parameters section below.

`python main.py --save_name=test_model --world=Hockey1v1`

# Viewing Results

Trained models (or the initially untrained models) can be viewed by setting the `render` flag when calling the main script. A `initial_model` needs to also be defined as the name of a model saved in the `/trained_models` folder.

`python main.py --initial_model=sample_model --world=Hockey1v1 --render=True`

# parameters

Coming soon.
