#!/usr/bin/env python3

import typer

app = typer.Typer()


@app.command()
def train_dql(flags: bool = False, save: bool = True):
    """Trains the DQL model

    Args:
        flags (bool): Wheter to allow flags.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from minesweeper_solver.DQL import DQL

    batch_size = 128
    episodes = 1000

    def epsilon(step: int):
        start = 0.9
        end = 0.1
        tc = 1000
        return end + (start - end) * np.exp(-step / tc)

    gamma = 0.99
    lr = 0.0003
    w_update_interval = 300

    env_args = [9, 9, 10]
    env_kwargs = {}

    agent = DQL(
        episodes,
        batch_size,
        epsilon,
        gamma,
        lr,
        w_update_interval,
        flags,
        save,
        env_args=env_args,
        env_kwargs=env_kwargs,
    )

    agent.train()

    plt.show()  # This leaves the training plot visible after its complete


@app.command()
def run_dql():
    """Runs the trained DQL model solver"""
    from minesweeper_solver.solver import SolverDQL

    solver = SolverDQL(width=9, height=9, mines=10)
    time = solver.run(1000)
    print(f"Time: {time:.4f} s")
    solver.quit()


@app.command()
def run_random():
    """Runs the random solver"""
    from minesweeper_solver.solver import SolverRandom

    solver = SolverRandom()
    time = solver.run(1000)
    print(f"Time: {time:.4f} s")
    solver.quit()


@app.command()
def run_naive():
    """Runs the naive solver"""
    from minesweeper_solver.solver import SolverNaive

    solver = SolverNaive()
    time = solver.run(1000)
    print(f"Time: {time:.4f} s")
    solver.quit()


if __name__ == "__main__":
    app()
