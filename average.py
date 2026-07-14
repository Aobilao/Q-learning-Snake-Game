from agent import HEIGHT, WIDTH, Agent, evaluate_avg
from game import Game

if __name__ == "__main__":
    game = Game(HEIGHT, WIDTH)
    agent = Agent()
    agent.load_values("values/augmented_values.pkl")
    avg = evaluate_avg(agent, game)
    print(avg)
