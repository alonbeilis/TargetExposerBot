import random

SIZE = 5


def create_frame(agent: int, targets: list, size: int):
    mapping = {target: 'target' for target in targets}
    mapping.update({agent: 'agent'})
    return [mapping.get(pos, random.randint(0, 1)) for pos in range(size*size)]


def frames_gen(agent: int, targets: list, time: int, size: int = SIZE):
    for t in range(time):
        yield create_frame(agent=agent, targets=targets, size=size)
