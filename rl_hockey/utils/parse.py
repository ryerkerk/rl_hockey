import argparse

def parse_arg():
    parser = argparse.ArgumentParser(usage="python %(prog)s --save_name SAVE_NAME [options]",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters
    # Switch save_name to required=True once finished testing
    parser.add_argument("--save_name", type=str, default="no-name", metavar=None,
                        help="File name for trained model and results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")

    parser.add_argument("--world", type=str, default="Hockey1v1Heuristic",
                        help="World to train in")
    parser.add_argument("--algorithm", type=str, default="double_dqn",
                        help="Reinforcement algorithm to use")
    parser.add_argument("--network", type=str, default="dueling_feed_forward",
                        help="Network architecture to use")
    parser.add_argument("--optimizer", type=str, default="adagrad",
                        help="Optimizer type")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate")
    parser.add_argument("--memory", type=str, default="prioritized",
                        help="Memory buffer to use")
    parser.add_argument("--initial_model", type=str, default="none",
                        help="Model to load, set to 'none' to begin from scratch")
    parser.add_argument("--total_frames", type=float, default=20e6,
                        help="Total number of frames to run")
    parser.add_argument("--total_episodes", type=float, default=60000,
                        help="Total number of frames to run")
    parser.add_argument("--train_frames", type=int, default=8,
                        help="Number of frames per training step")


    parser.add_argument("--beta", type=float, default=0.5,
                        help="Beta value used to prioritized memory")

    parser.add_argument("--render", type=bool, default=False,
                        help="Set to true to render results. Will not train in this case")
    parser.add_argument("--draw_scale", type=float, default=0.5,
                        help="Scale at which to render the world")
    parser.add_argument("--eps_end", type=float, default=0.05,
                        help="Final epsilon value")
    parser.add_argument("--eps_decay", type=float, default=500000,
                        help="Epsilon decay factor")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discounted reward rate")
    parser.add_argument("--hn_size", type=int, default=512,
                        help="Number of nodes per hidden layer")
    parser.add_argument("--mem_capacity", type=float, default=100000,
                        help="Capacity of memory buffer")
    params = vars(parser.parse_args())

    return params