#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from minesweeper_solver.DQL import DQL
from minesweeper_solver.utils import get_gamestate


def main():

    env_args = []  # [9, 9, 10]

    batch_size = 128
    episodes = 150

    def epsilon(step: int):
        start = 0.9
        end = 0.01
        tc = 1000
        return end + (start - end) * np.exp(-step / tc)

    gamma = 0.99
    lr = 0.0003
    w_update_interval = 300

    state = get_gamestate(1)

    agent = DQL(episodes, batch_size, epsilon, gamma, lr, w_update_interval, state=state, env_args=env_args)

    agent.train()

    plt.show()  # This leaves the training plot visible after its complete


if __name__ == "__main__":
    main()
