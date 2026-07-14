import random
from collections import deque

STRAIGHT, TURN_LEFT, TURN_RIGHT = 0, -1, 1

UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class Game:
    def __init__(self, height: int = 15, width: int = 17) -> None:
        self.height = height
        self.width = width
        self.reset()

    def reset(self) -> None:
        self.body = deque(
            [
                (self.height // 2, self.width // 2),
                (self.height // 2, self.width // 2 - 1),
                (self.height // 2, self.width // 2 - 2),
            ]
        )
        self.body_set = set(self.body)
        self.dir_idx = 1
        self.is_alive = True
        self._place_food()
        self.steps = 0
        self.death_cause: str | None = None
        self.game_won = False
        self.ate = False

    @property
    def score(self) -> int:
        return len(self.body) - 3

    def new_head(self, action: int) -> tuple[int, int]:
        head_i, head_j = self.body[0]
        dir_idx = (self.dir_idx + action) % 4
        d_i, d_j = DIRECTIONS[dir_idx]
        return (head_i + d_i, head_j + d_j)

    def in_bound(self, head: tuple[int, int]) -> bool:
        head_i, head_j = head
        return 0 <= head_i < self.height and 0 <= head_j < self.width

    def is_occupied(self, point: tuple[int, int]) -> bool:
        if not self.in_bound(point):
            return True
        return point in self.body_set and point != self.body[-1]

    def step(self, action: int) -> None:
        self.ate = False
        self.steps += 1
        new_head = self.new_head(action)
        self.dir_idx = (self.dir_idx + action) % 4

        if new_head == self.food_pos:
            self.body.appendleft(new_head)
            self.body_set.add(new_head)
            self._place_food()
            self.ate = True
            return

        tail = self.body[-1]
        if new_head != tail and new_head in self.body_set:
            self.is_alive = False
            self.death_cause = "body"
            return

        if not self.in_bound(new_head):
            self.is_alive = False
            self.death_cause = "wall"
            return

        self.body.appendleft(new_head)
        self.body.pop()
        self.body_set.remove(tail)
        self.body_set.add(new_head)

        if len(self.body) == self.width * self.height:
            self.game_won = True

    def _place_food(self) -> None:
        free = [
            (i, j)
            for i in range(self.height)
            for j in range(self.width)
            if (i, j) not in self.body_set
        ]
        if free:
            self.food_pos = random.choice(free)

    def play(self) -> None:
        while self.is_alive and not self.game_won:
            self.render()
            action = input("Choose action (or 'q' to quit): ")
            if action == "q":
                return
            action = int(action)
            self.step(action)
        if self.game_won:
            print("You won the game")
        else:
            print("You lost the game")
        self.reset()

    def render(self) -> None:
        RESET = "\033[0m"
        BG_GREEN = "\033[42m"
        BG_BGREEN = "\033[102m"
        BG_RED = "\033[41m"
        BG_WHITE = "\033[47m"
        BLOCK = "  "

        w = self.width + 2
        h = self.height + 2

        display = [[BLOCK for _ in range(w)] for _ in range(h)]

        for x in range(w):
            display[0][x] = f"{BG_WHITE}{BLOCK}{RESET}"
            display[h - 1][x] = f"{BG_WHITE}{BLOCK}{RESET}"
        for y in range(h):
            display[y][0] = f"{BG_WHITE}{BLOCK}{RESET}"
            display[y][w - 1] = f"{BG_WHITE}{BLOCK}{RESET}"

        for i, j in list(self.body)[1:]:
            display[i + 1][j + 1] = f"{BG_GREEN}{BLOCK}{RESET}"

        head_i, head_j = self.body[0]
        display[head_i + 1][head_j + 1] = f"{BG_BGREEN}{BLOCK}{RESET}"

        if self.food_pos:
            food_i, food_j = self.food_pos
            display[food_i + 1][food_j + 1] = f"{BG_RED}{BLOCK}{RESET}"

        print("\n".join("".join(row) for row in display))
