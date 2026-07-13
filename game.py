import random
from collections import deque

STRAIGHT, TURN_LEFT, TURN_RIGHT = 0, -1, 1

UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


class Game:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
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

    def new_head(self, action: int) -> tuple[int, int]:
        head_i, head_j = self.body[0]
        dir_idx = (self.dir_idx + action) % 4
        d_i, d_j = DIRECTIONS[dir_idx]
        new_i, new_j = head_i + d_i, head_j + d_j
        return (new_i, new_j)

    def in_bound(self, head: tuple[int, int]) -> bool:
        head_i, head_j = head
        return 0 <= head_i < self.height and 0 <= head_j < self.width

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
        if new_head in self.body_set - {tail}:
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
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for i, j in list(self.body)[1:]:
            grid[i][j] = "o"
        head_i, head_j = self.body[0]
        grid[head_i][head_j] = "H"
        if self.food_pos:
            food_i, food_j = self.food_pos
            grid[food_i][food_j] = "*"
        print("\n".join("".join(row) for row in grid))


if __name__ == "__main__":
    game = Game(15, 17)
    game.play()
