#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import numpy as np
from minesweeper_solver.DQL import DQL


def main():

    env_args = []  # [9, 9, 10]

    batch_size = 128
    episodes = 8

    def epsilon(step: int):
        start = 0.9
        end = 0.01
        tc = 1000
        return end + (start - end) * np.exp(-step / tc)

    gamma = 0.99
    lr = 0.0003
    w_update_interval = 300

    with open(os.path.join("data", "game_state.txt"), "r") as f:
        state = f.read()

    agent = DQL(episodes, batch_size, epsilon, gamma, lr, w_update_interval, state=state, env_args=env_args)

    try:
        a = agent.run()
    except Exception as e:
        agent._env.close_env()
        raise e
    agent._env.close_env()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
