import random
import pickle
import time
from game import DIRECTIONS, Game, STRAIGHT, TURN_LEFT, TURN_RIGHT
from typing import TypedDict, NotRequired

SAVE_PATH = "values/local_values.pkl"
VALUES_PATH = "values/values.pkl"
TRAINING_LOG = "values/augmented_training_log.pkl"
TURN_CHOICES = (TURN_LEFT, STRAIGHT, TURN_RIGHT)

HEIGHT = 15
WIDTH = 17
TRAINING_EPISODES = 1000000
DEFAULT_VALUE = 0.0
FOOD_REWARD = 10.0
LOSE_REWARD = -10.0
TIMEOUT_REWARD = -10.0
STEP_REWARD = -0.01
MAX_STEPS_TRAINING = HEIGHT * WIDTH
MAX_STEPS_PLAYING = 2000

State = tuple[bool, bool, bool, int, int, int, int, int, int]


class TrainingLog(TypedDict):
    score: list[int]
    steps: list[int]
    epsilon: NotRequired[list[float]]
    death_cause: list[str | None]
    states_visited: NotRequired[list[int]]


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

    def get_augmented_state(self, game: Game) -> State:
        is_occupied = game.is_occupied
        head_i, head_j = game.body[0]
        dir_idx = game.dir_idx

        danger: list[bool] = []
        rays: list[int] = []
        for turn in TURN_CHOICES:
            d_i, d_j = DIRECTIONS[(dir_idx + turn) % 4]
            ray = 4
            for dist in range(1, 4):
                if is_occupied((head_i + d_i * dist, head_j + d_j * dist)):
                    ray = dist
                    break
            danger.append(ray == 1)
            rays.append(ray)

        food_i, food_j = game.food_pos
        food_di = sign(food_i - head_i)
        food_dj = sign(food_j - head_j)

        return (*danger, dir_idx, food_di, food_dj, *rays)

    def Q(self, state: State) -> list[float]:
        q = self.values_dict.get(state)
        if q is None:
            q = [DEFAULT_VALUE] * 3
            self.values_dict[state] = q
        return q

    def _greedy_action(self, state: State) -> int:
        q = self.Q(state)
        best_val = float("-inf")
        best_turns: list[int] = []
        for turn in TURN_CHOICES:
            val = q[turn]
            if val > best_val:
                best_val = val
                best_turns = [turn]
            elif val == best_val:
                best_turns.append(turn)
        return random.choice(best_turns)

    def choose_action_from_state(self, state: State, epsilon: float) -> int:
        if random.random() <= epsilon:
            return random.choice(TURN_CHOICES)
        return self._greedy_action(state)

    def choose_action(self, game: Game, epsilon: float) -> int:
        return self.choose_action_from_state(self.get_augmented_state(game), epsilon)

    def compute_reward(self, game: Game) -> float:
        if not game.is_alive:
            return LOSE_REWARD
        if game.ate is True:
            return FOOD_REWARD
        if game.steps >= MAX_STEPS_TRAINING:
            return TIMEOUT_REWARD
        return STEP_REWARD

    def train(self, episodes: int, game: Game) -> TrainingLog:
        log: TrainingLog = {
            "score": [],
            "steps": [],
            "epsilon": [],
            "death_cause": [],
            "states_visited": [],
        }
        get_state = self.get_augmented_state
        Q = self.Q
        choose = self.choose_action_from_state
        compute_reward = self.compute_reward
        alpha = self.alpha
        gamma = self.gamma
        epsilon_end = self.epsilon_end
        epsilon_start = self.epsilon_start
        decay_rate = self.decay_rate
        values_dict = self.values_dict

        for episode in range(episodes):
            game.reset()
            epsilon = max(epsilon_end, epsilon_start * (decay_rate**episode))
            state = get_state(game)
            while (
                game.is_alive and not game.game_won and game.steps <= MAX_STEPS_TRAINING
            ):
                action = choose(state, epsilon)
                game.step(action)
                next_state = get_state(game)
                reward = compute_reward(game)

                is_terminal = (
                    not game.is_alive
                    or game.game_won
                    or game.steps > MAX_STEPS_TRAINING
                )
                q_max = 0.0 if is_terminal else max(Q(next_state))

                q_prev = Q(state)
                q_prev[action] += alpha * (reward + gamma * q_max - q_prev[action])

                state = next_state

            log["score"].append(game.score)
            log["steps"].append(game.steps)
            log["epsilon"].append(epsilon)
            log["death_cause"].append(game.death_cause)
            log["states_visited"].append(len(values_dict))

            print(f"\rFinished training {episode + 1} episodes", end="", flush=True)

        print()
        return log

    def play_games(self, game: Game, episodes: int) -> TrainingLog:
        log: TrainingLog = {
            "score": [],
            "steps": [],
            "death_cause": [],
        }
        for i in range(episodes):
            game.reset()
            self.play(game)

            log["score"].append(game.score)
            log["steps"].append(game.steps)
            log["death_cause"].append(game.death_cause)

            print(f"\rPlayed {i + 1} rounds", end="", flush=True)
            print()

        return log

    def save_values(self, path: str = SAVE_PATH) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.values_dict, file)
        print(f"Saved values to {path}")

    def load_values(self, path: str = VALUES_PATH) -> None:
        with open(path, "rb") as file:
            self.values_dict = pickle.load(file)
        print(f"Loaded values from {path}")

    def play(self, game: Game) -> int:
        game.reset()
        while game.is_alive and not game.game_won and game.steps <= MAX_STEPS_PLAYING:
            action = self.choose_action(game, epsilon=0.0)
            game.step(action)
        return game.score


def watch(agent: Agent, game: Game, delay: float = 0.1) -> None:
    game.reset()
    while game.is_alive and not game.game_won:
        print("\033[H\033[J", end="")
        game.render()
        print(f"Score: {game.score}  Steps: {game.steps}")
        action = agent.choose_action(game, epsilon=0.0)
        game.step(action)
        time.sleep(delay)
    print("\033[H\033[J", end="")
    game.render()
    print(
        f"Game over! Score: {game.score}  Steps: {game.steps}  Death: {game.death_cause}"
    )


def evaluate_avg(agent: Agent, game: Game, episodes: int = 10000) -> float:
    avg = 0
    for i in range(episodes):
        score = agent.play(game)
        avg += (score - avg) / (i + 1)
        print(f"\rFinished {i + 1} episodes", end="", flush=True)
    print()
    return avg


def save_log(log: TrainingLog, path: str = TRAINING_LOG) -> None:
    with open(path, "wb") as file:
        pickle.dump(log, file)
    print(f"Saved log to {path}")


def load_log(path: str = TRAINING_LOG) -> TrainingLog:
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    game = Game(HEIGHT, WIDTH)
    agent = Agent()

    target_episode = TRAINING_EPISODES * 0.6
    decay_rate = (agent.epsilon_end / agent.epsilon_start) ** (1 / target_episode)
    agent.decay_rate = decay_rate

    log = agent.train(TRAINING_EPISODES, game)
    agent.save_values()
    save_log(log, TRAINING_LOG)

    avg = evaluate_avg(agent, game)
    print(avg)
