#!/usr/bin/env python3

import os
import typer
from typing import Annotated, Any
from .utils import get_gamestate


def run_solver(solver: Any, tries: int):
    try:
        time = solver.run(tries)
        if time:
            print(f"Time: {time:.4f} s")
    finally:
        solver.quit()


app = typer.Typer()


episodes_type = Annotated[int, typer.Argument(help="The number of episodes to perform in the training.")]
flags_type = Annotated[bool, typer.Option(help="Whether the model uses flags.")]
save_type = Annotated[bool, typer.Option(help="Whether to save the model.")]


@app.command()
def train_dql(episodes: episodes_type = 5000, flags: flags_type = False, save: save_type = True):
    """Trains the DQL model"""

    import matplotlib.pyplot as plt
    import numpy as np
    from minesweeper_solver.DQL import DQL

    batch_size = 128

    def epsilon(step: int) -> float:
        start = 0.9
        end = 0.1
        tc = 1000
        return end + (start - end) * np.exp(-step / tc)

    gamma = 0.99
    lr = 0.0003
    w_update_interval = 300

    env_args = []  # [9, 9, 10]
    env_kwargs = {"state": get_gamestate(1)}

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


model_filepath_type = Annotated[str, typer.Argument(help="The filepath for the model weigths to use.")]
tries_type = Annotated[int, typer.Argument(help="The amount of games to play at max.")]


@app.command()
def run_dql(
    model_filepath: model_filepath_type = os.path.join("models", "model_no_flags.pt"),
    tries: tries_type = 1,
):
    """Runs the trained DQL model solver"""
    from minesweeper_solver.solver import SolverDQL

    solver = SolverDQL(model_filepath, width=9, height=9, mines=10)
    run_solver(solver, tries)


@app.command()
def run_random(tries: tries_type = 1):
    """Runs the random solver"""
    from minesweeper_solver.solver import SolverRandom

    solver = SolverRandom()
    run_solver(solver, tries)


@app.command()
def run_naive(tries: tries_type = 1):
    """Runs the naive solver"""
    from minesweeper_solver.solver import SolverNaive

    solver = SolverNaive()
    run_solver(solver, tries)


if __name__ == "__main__":
    app()
