from game import Game
from agent import Agent, watch

if __name__ == "__main__":
    game = Game(15, 17)
    agent = Agent()
    agent.load_values("augmented_values.pkl")
    watch(agent, game)
