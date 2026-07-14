import random
import pickle
import time
from game import Game, STRAIGHT, TURN_LEFT, TURN_RIGHT

SAVE_PATH = "local_values.pkl"
VALUES_PATH = "values.pkl"
TURN_CHOICES = (TURN_LEFT, STRAIGHT, TURN_RIGHT)

HEIGHT = 15
WIDTH = 17
TRAINING_EPISODES = 1000000
DEFAULT_VALUE = 0.0
FOOD_REWARD = 10.0
LOSE_REWARD = -10.0
TIMEOUT_REWARD = -10.0
STEP_REWARD = -0.01
MAX_STEPS = 17 * 15

State = tuple[bool, bool, bool, int, int, int]


def sign(x):
    return (x > 0) - (x < 0)


class Agent:
    def __init__(
        self,
        gamma: float = 0.95,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        decay_rate: float = 0.995,
    ) -> None:
        self.values_dict: dict[State, list[float]] = {}
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate

    def get_state(self, game: Game) -> State:
        tail = game.body[-1]
        dir_idx = game.dir_idx
        danger: list[bool] = []
        for turn in TURN_CHOICES:
            new_head = game.new_head(turn)
            danger.append(
                new_head in game.body_set - {tail} or not game.in_bound(new_head)
            )

        food_i, food_j = game.food_pos
        head_i, head_j = game.body[0]
        food_di = sign(food_i - head_i)
        food_dj = sign(food_j - head_j)

        return (*danger, dir_idx, food_di, food_dj)

    def Q(self, state: State) -> list[float]:
        return self.values_dict.setdefault(state, [DEFAULT_VALUE] * 3)

    def choose_action(self, game: Game, epsilon: float) -> int:
        if random.random() <= epsilon:
            return random.choice(TURN_CHOICES)
        state = self.get_state(game)
        best_val = float("-inf")
        best_turns = []
        for turn in TURN_CHOICES:
            val = self.Q(state)[turn]
            if val > best_val:
                best_val = val
                best_turns = [turn]
            elif val == best_val:
                best_turns.append(turn)
        return random.choice(best_turns)

    def compute_reward(self, game: Game) -> float:
        if not game.is_alive:
            return LOSE_REWARD
        if game.ate is True:
            return FOOD_REWARD
        if game.steps >= MAX_STEPS:
            return TIMEOUT_REWARD
        return STEP_REWARD

    def train(self, episodes: int, game: Game) -> None:
        for episode in range(episodes):
            game.reset()
            while game.is_alive and not game.game_won and game.steps <= MAX_STEPS:
                previous_state = self.get_state(game)
                epsilon = max(
                    self.epsilon_end, self.epsilon_start * (self.decay_rate**episode)
                )
                action = self.choose_action(game, epsilon)
                game.step(action)
                next_state = self.get_state(game)
                reward = self.compute_reward(game)

                is_terminal = (
                    not game.is_alive or game.game_won or game.steps > MAX_STEPS
                )
                Q_max = max(self.Q(next_state)) if not is_terminal else 0.0

                self.Q(previous_state)[action] += self.alpha * (
                    reward + self.gamma * Q_max - self.Q(previous_state)[action]
                )
            print(f"\rFinished training {episode + 1} episodes", end="", flush=True)

    def save(self, path: str = SAVE_PATH) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.values_dict, file)
        print(f"Saved data to {path}")

    def load(self, path: str = VALUES_PATH) -> None:
        with open(path, "rb") as file:
            self.values_dict = pickle.load(file)
        print(f"Loaded data from {path}")

    def play(self, game: Game) -> int:
        game.reset()
        while game.is_alive and not game.game_won:
            action = self.choose_action(game, epsilon=0.0)
            game.step(action)
        return len(game.body) - 3


def watch(agent: Agent, game: Game, delay: float = 0.1) -> None:
    game.reset()
    while game.is_alive and not game.game_won:
        print("\033[H\033[J", end="")
        game.render()
        print(f"Score: {len(game.body) - 3}  Steps: {game.steps}")
        action = agent.choose_action(game, epsilon=0.0)
        game.step(action)
        time.sleep(delay)
    print("\033[H\033[J", end="")
    game.render()
    print(
        f"Game over! Score: {len(game.body) - 3}  Steps: {game.steps}  Death: {game.death_cause}"
    )


def evaluate_avg(episodes: int, agent: Agent, game: Game) -> float:
    avg = 0
    for i in range(episodes):
        score = agent.play(game)
        avg += (score - avg) / (i + 1)
        print(f"\rFinished {i + 1} episodes", end="", flush=True)
    print()
    return avg


if __name__ == "__main__":
    game = Game(HEIGHT, WIDTH)
    target_episode = TRAINING_EPISODES * 0.6
    decay_rate = (0.01 / 1.0) ** (1 / target_episode)
    agent = Agent(decay_rate=decay_rate)
    agent.train(TRAINING_EPISODES, game)
    agent.save()
    avg = evaluate_avg(10000, agent, game)
    print(avg)
