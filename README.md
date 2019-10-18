![](https://raw.githubusercontent.com/ryerkerk/rl_hockey/master/sample_view.gif)

# rl_hockey
An exploration of applying reinforcement learning to an air hockey style game. 

This project includes a simple rigid body physics engine, on top of which several variations of the game, or worlds, are built. These act as testbeds for several reinforcement learning models and algorithms.

This project requires `numpy` and `pytorch`.

# Worlds

The desired world is set in `main.py`.

The 2v2 worlds remain a work in progress. They can be trained, but fail to produce the expected behavior. 

- **Hockey1v1**: This is a 1v1 game where the controller is trained through self play.
- **Hockey1v1Heuristic**: The controller is trained by competing against a very basic bot.
- **Hockey2v2**: This is a 2v2 game. Only a single controller is used/trained, each agent is controlled by separate instance of this controller. Generally I've found that both agents on a team will remain clumped together at all times.
- **Hockey2v2Roles**: In this 2v2 game two separate controllers are used, one for a defensive agent and one for an offensive agent. The two controllers differ only in how scores are assigned to each agent. I've found that in most cases one of the two agents will come to dominate the game, while the second agent exhibits no meaningful behavior.

# Training

To train a model you must set `VIEW_RUN = False` in `main.py`.

A number of parameters for the network architecture and training algorithm can be set in `main.py`. DQN and double DQN algorithms, feed forward and dueling feed foward network architectures, and basic, prioritized, or impact prioritized memories are supported. 

Impact priortized memory replaces old memories based on age and a measure of impact. At the moment impact is measured as the distance of the agent to the puck. I've found this can reduce the required learning time, but may also reduce model stability.

Most worlds use self play for training. Each iteration the controller will compete against a previous screenshot of itself. Every 1000 iterations a screenshot of the current controller(s) is taken, with the 5 most recently screenshots kept in memory.

Every 100 iterations the current model will be saved to the filename specified by `SAVE_NAME`. Restarting training is currently not supported.

# Viewing

Once a model is trained, it can be viewed by setting `VIEW_RUN = True`, setting `PREVIOUS_MODEL` to the point to the trained model, and ensuring that `WORLD` is set to the world on which the model was trained. Both teams will user the same model.
