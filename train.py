from agent import Agent
from game import Game


def update(agent: Agent, game: Game, epsilon: float):
    if random.random() <= epsilon:
        action = random.choice((-1, 0, 1))
