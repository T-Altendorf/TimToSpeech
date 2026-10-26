"""The night lane's list: ordered texts and each one's state, on disk.

Saved as `night-queue.json` under CACHE_DIR, written atomically (a temporary
file beside it, then a rename), so a redeploy resumes where it stopped. Each
text has one state letter in `states`, a string as long as `texts`:

- `.` still to do
- `d` done (synthesised by the night lane)
- `c` skipped: a clip of the current audio version was already there
- `l` skipped: longer than the free endpoint takes (never sent to paid)
- `f` failed (an error that was neither a 429 nor a refusal)
"""
import json
import os
import threading
from pathlib import Path

from .config import log

PENDING, DONE, CACHED, TOO_LONG, FAILED = ".", "d", "c", "l", "f"
APPEND, REPLACE = "append", "replace"


class NightList:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.texts: list = []
        self.states: list = []
        self.listed: set = set()
        self.cursor = 0
        self.generation = 0  # bumped on replace, so a mark for an old list is dropped
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
            texts, states = list(data["texts"]), list(data["states"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            log(f"✗ Night list at {self.path.name} unreadable, starting empty: {error}")
            return
        if len(texts) != len(states):
            log(f"✗ Night list at {self.path.name} has mismatched states, starting empty")
            return
        self.texts, self.states, self.listed = texts, states, set(texts)

    def _save(self) -> None:
        """Called with the lock held."""
        body = json.dumps({"texts": self.texts, "states": "".join(self.states)}, ensure_ascii=False)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(body, "utf-8")
        os.replace(tmp, self.path)

    def add(self, texts: list, mode: str, is_cached) -> dict:
        """Lists the texts not listed nor cached; counts each kind."""
        wanted = list(dict.fromkeys(text for text in texts if text))
        with self.lock:
            if mode == REPLACE:
                self.texts, self.states, self.listed = [], [], set()
                self.cursor, self.generation = 0, self.generation + 1
            counts = {"listed": len(wanted), "added": 0, "already_listed": 0, "already_cached": 0}
            for text in wanted:
                if text in self.listed:
                    counts["already_listed"] += 1
                elif is_cached(text):
                    counts["already_cached"] += 1
                else:
                    self.texts.append(text)
                    self.states.append(PENDING)
                    self.listed.add(text)
                    counts["added"] += 1
            self._save()
            counts["total"] = len(self.texts)
            return counts

    def next_pending(self) -> tuple | None:
        """(generation, index, text) of the first text still to do."""
        with self.lock:
            while self.cursor < len(self.states) and self.states[self.cursor] != PENDING:
                self.cursor += 1
            if self.cursor == len(self.states):
                return None
            return self.generation, self.cursor, self.texts[self.cursor]

    def mark(self, generation: int, index: int, state: str) -> None:
        with self.lock:
            if generation != self.generation:
                return
            self.states[index] = state
            self._save()

    def counts(self) -> dict:
        with self.lock:
            states = "".join(self.states)
        skipped = states.count(CACHED) + states.count(TOO_LONG)
        return {
            "listed": len(states),
            "done": states.count(DONE),
            "skipped": skipped,
            "skipped_reasons": {"cached": states.count(CACHED), "too_long": states.count(TOO_LONG)},
            "failed": states.count(FAILED),
            "remaining": states.count(PENDING),
        }
