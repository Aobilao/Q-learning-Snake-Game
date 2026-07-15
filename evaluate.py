import random

from agent import HEIGHT, SEED, WIDTH, Agent, save_log
from game import Game

if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    game = Game(HEIGHT, WIDTH)
    agent = Agent()
    agent.load_values("values/local_values.pkl")
    log = agent.play_games(game, 100000)
    save_log(log, "values/local_run_log.pkl")
