import os


def get_gamestate(number: int):
    """Returns the corresponding saved gamestate form the resources -folder."""
    with open(os.path.join("tests", "resources", f"game_state_{number}.txt"), "r") as f:
        return f.read()
