# Minesweeper-Solver

Contains solvers and a framework for easily building new ones. The SolverBase -class takes care of
starting a new game and handling the interactions with it. At its simplest, a new solver can just
inherit from the base class and implement only the `_run()` method, that contains the solver loop.


## Usage

Clone the repository and run
```
pip install .
```
to install the dependencies. Also follow [these](https://pytorch.org/get-started/locally/) instructions to install pytorch.

Run the command
```
msolver --help
```
for information on the available commands, and
```
msolver command --help
```
for information on individual commands.

## Solvers

Currently there are three solvers:

### Random:

This solver randomly clicks on cells.

### Naive:

This solver makes decisions based on 3x3 subgrids. If it determines that all the unopened cells
in a 3x3 subgrid are mines it flags them, if it thinks that all of them are safe, it opens them.
If it can't deduct anymore safe or flaggable cells, it opens the lowest probability cell in the grid.

### DQL:

This solver is is powered by a reinforcement learning model. DQL is used to train a model to select
the safest cell to open.
