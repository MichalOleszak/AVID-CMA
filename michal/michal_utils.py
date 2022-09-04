from sys import platform


def is_running_on_polyaxon():
    return False if platform == "darwin" else True

