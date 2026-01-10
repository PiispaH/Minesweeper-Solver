#!/usr/bin/env python3

import os
from typing import Annotated, Any
import typer


def run_solver(solver: Any, tries: int):
    try:
        time = solver.run(tries)
        if time:
            print(f"Time: {time:.4f} s")
    finally:
        solver.quit()


app = typer.Typer(pretty_exceptions_show_locals=False)


episodes_type = Annotated[int, typer.Argument(help="The number of episodes to perform in the training.")]
flags_type = Annotated[bool, typer.Option(help="Whether the model uses flags.")]
save_type = Annotated[bool, typer.Option(help="Whether to save the model.")]


@app.command()
def train_dql(episodes: episodes_type = 5000, flags: flags_type = False, save: save_type = True):
    """Trains the DQL model"""

    try:
        import torch
    except ImportError as e:
        print("PyTorch not installed")
        return
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

    env_args = [9, 9, 10]

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
    )

    agent.train()

    plt.show()  # This leaves the training plot visible after its complete


model_filepath_type = Annotated[str, typer.Argument(help="The filepath for the model weigths to use.")]
tries_type = Annotated[int, typer.Option(help="The amount of games to play at max.")]


@app.command()
def run_dql(
    model_filepath: model_filepath_type = os.path.join("models", "model_no_flags.pt"),
    tries: tries_type = 5,
):
    """Runs the trained DQL model solver"""
    try:
        import torch
    except ImportError as e:
        print("PyTorch not installed")
        return
    from minesweeper_solver.dql_solver import SolverDQL

    solver = SolverDQL(9, 9, 10, tries, 0.1, filepath=model_filepath)
    solver.run()


width_type = Annotated[int, typer.Argument(help="The width of the grid.", min=3, max=30, clamp=True)]
height_type = Annotated[int, typer.Argument(help="The height of the grid", min=3, max=16, clamp=True)]
n_mines_type = Annotated[int, typer.Argument(help="The amount of mines.", min=0, max=99, clamp=True)]
speed_type = Annotated[float, typer.Option(help="The dealay between every action taken.", min=0.0, max=1.0, clamp=True)]


@app.command()
def run_random(
    width: width_type = 30,
    height: height_type = 16,
    n_mines: n_mines_type = 99,
    tries: tries_type = 5,
    speed: speed_type = 0.3,
):
    """Runs the random solver"""
    from minesweeper_solver.solver import SolverRandom

    solver = SolverRandom(width, height, n_mines, tries, action_delay=speed)
    solver.run()


@app.command()
def run_naive(
    width: width_type = 30,
    height: height_type = 16,
    n_mines: n_mines_type = 99,
    tries: tries_type = 5,
    speed: speed_type = 0.3,
):
    """Runs the naive solver"""
    from minesweeper_solver.solver import SolverNaive

    solver = SolverNaive(width, height, n_mines, tries, action_delay=speed)
    solver.run()


if __name__ == "__main__":
    app()
