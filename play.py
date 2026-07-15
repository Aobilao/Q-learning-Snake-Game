from game import Game
from agent import HEIGHT, WIDTH, Agent, watch

if __name__ == "__main__":
    game = Game(HEIGHT, WIDTH)
    agent = Agent()
    agent.load_values("values/augmented_values.pkl")
    watch(agent, game, 0.1)
