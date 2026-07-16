import random
import pickle
import time

from game import DIRECTIONS, Game, STRAIGHT, TURN_LEFT, TURN_RIGHT
from typing import TypedDict

TURN_CHOICES = (TURN_LEFT, STRAIGHT, TURN_RIGHT)

SAVE_PATH = "values/local_values.pkl"
LOAD_PATH = "values/augmented_values.pkl"
TRAINING_LOG_PATH = "values/local_training_log.pkl"

SEED: int | None = 42
HEIGHT = 15
WIDTH = 17
TRAINING_EPISODES = 1000000
DEFAULT_VALUE = 0.0
FOOD_REWARD = 10.0
LOSE_REWARD = -10.0
TIMEOUT_REWARD = -10.0
STEP_REWARD = -0.01
TIMEOUT_STEPS_TRAINING = HEIGHT * WIDTH
TIMEOUT_STEPS_PLAYING = 2000

State = tuple[int, int, int, int, int, int, int, int]


class TrainingLog(TypedDict):
    score: list[int]
    steps: list[int]
    epsilon: list[float]
    death_cause: list[str | None]
    states_visited: list[int]


class EvalLog(TypedDict):
    score: list[int]
    steps: list[int]
    death_cause: list[str | None]


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
        height = game.height
        width = game.width
        body_set = game.body_set
        tail = game.body[-1]
        head_i, head_j = game.body[0]
        dir_idx = game.dir_idx

        rays: list[int] = []
        for turn in TURN_CHOICES:
            d_i, d_j = DIRECTIONS[(dir_idx + turn) % 4]
            ray = 4
            for dist in range(1, 4):
                p_i = head_i + d_i * dist
                p_j = head_j + d_j * dist
                if not (0 <= p_i < height and 0 <= p_j < width):
                    ray = dist
                    break
                point = (p_i, p_j)
                if point in body_set and point != tail:
                    ray = dist
                    break
            rays.append(ray)

        food_i, food_j = game.food_pos
        food_di = sign(food_i - head_i)
        food_dj = sign(food_j - head_j)

        tail_i, tail_j = tail
        tail_di = sign(tail_i - head_i)
        tail_dj = sign(tail_j - head_j)

        return (dir_idx, food_di, food_dj, *rays, tail_di, tail_dj)

    def Q(self, state: State) -> list[float]:
        q = self.values_dict.get(state)
        if q is None:
            q = [DEFAULT_VALUE] * 3
            self.values_dict[state] = q
        return q

    def _greedy_action_from_q(self, q: list[float]) -> int:
        best_val = float("-inf")
        best_turns: list[int] = []
        for turn in TURN_CHOICES:
            val = q[turn]
            if val > best_val:
                best_val = val
                best_turns = [turn]
            elif val == best_val:
                best_turns.append(turn)
        if len(best_turns) == 1:
            return best_turns[0]
        return random.choice(best_turns)

    def choose_action_greedy(self, game: Game) -> int:
        return self._greedy_action_from_q(self.Q(self.get_augmented_state(game)))

    def compute_reward(self, game: Game) -> float:
        if not game.is_alive:
            return LOSE_REWARD
        if game.ate is True:
            return FOOD_REWARD
        if game.steps_since_food >= TIMEOUT_STEPS_TRAINING:
            return TIMEOUT_REWARD
        return STEP_REWARD

    def train(self, game: Game, episodes: int) -> TrainingLog:
        start_time = time.perf_counter()
        log: TrainingLog = {
            "score": [],
            "steps": [],
            "epsilon": [],
            "death_cause": [],
            "states_visited": [],
        }
        get_state = self.get_augmented_state
        Q = self.Q
        greedy_from_q = self._greedy_action_from_q
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
            q_state = Q(get_state(game))
            while (
                game.is_alive
                and not game.game_won
                and game.steps_since_food < TIMEOUT_STEPS_TRAINING
            ):
                if random.random() < epsilon:
                    action = TURN_CHOICES[int(random.random() * 3)]
                else:
                    action = greedy_from_q(q_state)
                game.step(action)
                reward = compute_reward(game)

                is_terminal = (
                    not game.is_alive
                    or game.game_won
                    or game.steps_since_food >= TIMEOUT_STEPS_TRAINING
                )
                q_next = Q(get_state(game))
                q_max = 0.0 if is_terminal else max(q_next)

                q_state[action] += alpha * (reward + gamma * q_max - q_state[action])

                q_state = q_next

            death_cause = game.death_cause
            if death_cause is None and not game.game_won:
                death_cause = "timeout"

            log["score"].append(game.score)
            log["steps"].append(game.steps)
            log["epsilon"].append(epsilon)
            log["death_cause"].append(death_cause)
            log["states_visited"].append(len(values_dict))

            if (episode + 1) % max(1, episodes // 100) == 0:
                print(f"\rFinished training {episode + 1} episodes", end="", flush=True)

        elapsed = time.perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"\nTrained {episodes} episodes in {int(hours)}h {int(minutes)}m {seconds:.1f}s"
        )
        return log

    def play_games(self, game: Game, episodes: int) -> EvalLog:
        start_time = time.perf_counter()
        log: EvalLog = {
            "score": [],
            "steps": [],
            "death_cause": [],
        }
        for episode in range(episodes):
            self.play(game)
            death_cause = game.death_cause
            if death_cause is None and not game.game_won:
                death_cause = "timeout"

            log["score"].append(game.score)
            log["steps"].append(game.steps)
            log["death_cause"].append(death_cause)

            if (episode + 1) % max(1, episodes // 100) == 0:
                print(f"\rPlayed {episode + 1} rounds", end="", flush=True)

        elapsed = time.perf_counter() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f"\nTrained {episodes} episodes in {int(hours)}h {int(minutes)}m {seconds:.1f}s"
        )
        return log

    def save_values(self, path: str = SAVE_PATH) -> None:
        with open(path, "wb") as file:
            pickle.dump(self.values_dict, file)
        print(f"Saved values to {path}")

    def load_values(self, path: str = LOAD_PATH) -> None:
        with open(path, "rb") as file:
            self.values_dict = pickle.load(file)
        print(f"Loaded values from {path}")

    def play(self, game: Game) -> int:
        game.reset()
        while (
            game.is_alive
            and not game.game_won
            and game.steps_since_food < TIMEOUT_STEPS_PLAYING
        ):
            action = self.choose_action_greedy(game)
            game.step(action)
        return game.score


def watch(agent: Agent, game: Game, delay: float = 0.1) -> None:
    game.reset()
    while game.is_alive and not game.game_won:
        print("\033[H\033[J", end="")
        game.render()
        print(f"Score: {game.score}  Steps: {game.steps}")
        action = agent.choose_action_greedy(game)
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


def save_log(log: TrainingLog | EvalLog, path: str = TRAINING_LOG_PATH) -> None:
    with open(path, "wb") as file:
        pickle.dump(log, file)
    print(f"Saved log to {path}")


def load_log(path: str = TRAINING_LOG_PATH) -> TrainingLog:
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    game = Game(HEIGHT, WIDTH)
    agent = Agent()

    target_episode = TRAINING_EPISODES * 0.6
    decay_rate = (agent.epsilon_end / agent.epsilon_start) ** (1 / target_episode)
    agent.decay_rate = decay_rate

    log = agent.train(game, TRAINING_EPISODES)
    agent.save_values(SAVE_PATH)
    save_log(log, TRAINING_LOG_PATH)

    avg = evaluate_avg(agent, game)
    print(avg)
