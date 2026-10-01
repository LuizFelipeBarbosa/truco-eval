"""Re-drive a match from a logged engine event file and verify determinism."""

from __future__ import annotations

import json

from truco.match import TrucoMatch


def load_events(path: str) -> list[dict]:
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]


def replay_events(events: list[dict]) -> tuple[TrucoMatch, bool]:
  """Rebuild the match from seed + logged actions; return (match, identical)."""
  start = next(e for e in events if e["type"] == "match_start")
  m = TrucoMatch(start["seed"], start["num_players"], target=start.get("target", 12),
                 initial_scores=start.get("scores"))
  from truco.match import TALK_MAX_CHARS  # pylint: disable=import-outside-toplevel

  pending_truncation: dict | None = None
  for e in events:
    if e["type"] == "talk_truncated":
      # The engine logs this just before the action whose talk it truncated.
      pending_truncation = e
    elif e["type"] == "action":
      talk = e.get("talk")
      if talk and pending_truncation and pending_truncation["seat"] == e["seat"]:
        # Only the first TALK_MAX_CHARS survive in the log; pad back to the
        # original length so the engine re-emits an identical truncation event.
        talk = talk + "x" * (int(pending_truncation["original_length"]) - TALK_MAX_CHARS)
      pending_truncation = None
      m.apply_action(e["seat"], e["action"], talk)
  original = "".join(
      json.dumps({k: v for k, v in e.items() if k != "source"}, sort_keys=True,
                 ensure_ascii=False, separators=(",", ":")) + "\n"
      for e in events if e.get("source", "engine") == "engine"
  )
  return m, m.serialize_events() == original


def replay_file(path: str) -> tuple[TrucoMatch, bool]:
  return replay_events(load_events(path))
