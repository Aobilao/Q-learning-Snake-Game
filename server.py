import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import Agent
from bfs_agent import BFSAgent
from game import DIRECTIONS, Game, STRAIGHT, TURN_LEFT, TURN_RIGHT

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "web", "index.html")

AGENT_SPECS = {
    "rays": ("Rays", Agent, "values/augmented_values.pkl", "q"),
    "bfs": ("BFS", BFSAgent, "values/bfs_values.pkl", "bfs"),
}

DIR_INDEX = {"up": 0, "right": 1, "down": 2, "left": 3}
TURN_NAME = {TURN_LEFT: "left", STRAIGHT: "straight", TURN_RIGHT: "right"}


class Session:
    def __init__(self, initial: str) -> None:
        self.lock = threading.Lock()
        self.game = Game()
        self.agents = {}
        for name, (label, factory, path, kind) in AGENT_SPECS.items():
            inst = factory()
            inst.load_values(path)
            self.agents[name] = {"label": label, "agent": inst, "kind": kind}
        self.current = initial

    def _running(self) -> bool:
        return self.game.is_alive and not self.game.game_won

    def _policy(self):
        if not self._running():
            return None, None
        entry = self.agents[self.current]
        agent, kind = entry["agent"], entry["kind"]
        if kind == "bfs":
            canon, mirrored = agent.get_augmented_state(self.game)
            q = agent.values_dict.get(canon) or [0.0, 0.0, 0.0]
            left, straight, right = q[TURN_LEFT], q[STRAIGHT], q[TURN_RIGHT]
            if mirrored:
                left, right = right, left
        else:
            state = agent.get_augmented_state(self.game)
            q = agent.values_dict.get(state) or [0.0, 0.0, 0.0]
            left, straight, right = q[TURN_LEFT], q[STRAIGHT], q[TURN_RIGHT]
        vals = {"left": left, "straight": straight, "right": right}
        greedy = max(("straight", "left", "right"), key=lambda k: vals[k])
        return vals, greedy

    def state(self) -> dict:
        g = self.game
        entry = self.agents[self.current]
        vals, greedy = self._policy()
        return {
            "width": g.width,
            "height": g.height,
            "body": [list(cell) for cell in g.body],
            "food": list(g.food_pos) if g.food_pos else None,
            "dir_idx": g.dir_idx,
            "score": g.score,
            "steps": g.steps,
            "steps_since_food": g.steps_since_food,
            "alive": g.is_alive,
            "won": g.game_won,
            "death_cause": g.death_cause,
            "running": self._running(),
            "agent": self.current,
            "agent_label": entry["label"],
            "agents": [{"name": n, "label": e["label"]} for n, e in self.agents.items()],
            "states_visited": len(entry["agent"].values_dict),
            "policy": vals,
            "greedy": greedy,
        }

    def step(self) -> dict:
        with self.lock:
            if self._running():
                agent = self.agents[self.current]["agent"]
                self.game.step(agent.choose_action_greedy(self.game))
            return self.state()

    def move(self, direction: str) -> dict:
        with self.lock:
            if self._running() and direction in DIR_INDEX:
                delta = (DIR_INDEX[direction] - self.game.dir_idx) % 4
                action = {1: TURN_RIGHT, 3: TURN_LEFT}.get(delta, STRAIGHT)
                self.game.step(action)
            return self.state()

    def set_agent(self, name: str) -> dict:
        with self.lock:
            if name in self.agents:
                self.current = name
            return self.state()

    def reset(self) -> dict:
        with self.lock:
            self.game.reset()
            return self.state()

    def snapshot(self) -> dict:
        with self.lock:
            return self.state()


class Handler(BaseHTTPRequestHandler):
    session: Session

    def log_message(self, *args) -> None:
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def _file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._file(INDEX_PATH, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(self.session.snapshot())
        else:
            self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path == "/api/step":
            self._json(self.session.step())
        elif self.path == "/api/reset":
            self._json(self.session.reset())
        elif self.path == "/api/move":
            self._json(self.session.move(self._body().get("dir", "")))
        elif self.path == "/api/agent":
            self._json(self.session.set_agent(self._body().get("name", "")))
        else:
            self.send_error(404, "not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snake RL web UI")
    parser.add_argument("--agent", choices=list(AGENT_SPECS), default="rays")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    Handler.session = Session(args.agent)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving Snake UI at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
