"""Per-match logging: JSONL transcript, pure engine event log, text replay."""

from __future__ import annotations

import json
import os
from typing import Any, TextIO

from truco.match import TrucoMatch


def _dumps(obj: Any) -> str:
  return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class MatchLogger:
  """Writes ``transcript.jsonl`` (engine + LLM events), ``engine_events.jsonl``
  (engine only; byte-identical for the same seed and action sequence) and
  ``replay.txt`` (human-readable)."""

  def __init__(self, out_dir: str | None):
    self.out_dir = out_dir
    self._transcript: TextIO | None = None
    self._replay_lines: list[str] = []
    self._engine_cursor = 0
    self.transcript_events: list[dict[str, Any]] = []
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
      self._transcript = open(os.path.join(out_dir, "transcript.jsonl"), "w", encoding="utf-8")

  def event(self, event: dict[str, Any]) -> None:
    self.transcript_events.append(event)
    if self._transcript:
      self._transcript.write(_dumps(event) + "\n")

  def replay(self, line: str = "") -> None:
    self._replay_lines.append(line)

  def flush_engine_events(self, engine: TrucoMatch) -> None:
    """Copy engine events produced since the last flush into the transcript."""
    new = engine.events[self._engine_cursor:]
    self._engine_cursor = len(engine.events)
    for e in new:
      self.event({"source": "engine", **e})
      self._replay_engine_event(e)

  def close_transcript(self) -> None:
    """Close the transcript stream; safe to call more than once."""
    if self._transcript:
      self._transcript.close()
      self._transcript = None

  def _replay_engine_event(self, e: dict[str, Any]) -> None:
    t = e["type"]
    if t == "hand_start":
      self.replay()
      self.replay(f"--- Hand #{e['hand'] + 1} ({e['hand_type']}) — dealer seat {e['dealer']}, "
                  f"mão seat {e['mao']}, vira {e['vira']} (manilha {e['manilha_rank']}), "
                  f"score A {e['scores']['A']} – B {e['scores']['B']}")
      for seat, cards in e["hands"].items():
        self.replay(f"    seat {seat} dealt: {' '.join(cards)}")
    elif t == "mao_de_onze":
      self.replay(f"    MÃO DE ONZE for Team {e['team']}; seat {e['decider']} decides")
    elif t == "mao_de_ferro":
      self.replay("    MÃO DE FERRO: blind hand for 1 point")
    elif t == "action":
      talk = f'  — says: "{e["talk"]}"' if e.get("talk") else ""
      self.replay(f"  seat {e['seat']}: {e['action']}{talk}")
    elif t == "talk_truncated":
      self.replay(f"    (talk from seat {e['seat']} truncated from {e['original_length']} chars)")
    elif t == "trick_result":
      plays = ", ".join(f"{p['seat']}:{p['card']}" for p in e["plays"])
      res = "PARDA" if e["winner_team"] is None else f"Team {e['winner_team']} (seat {e['winner_seat']})"
      self.replay(f"    trick {e['trick'] + 1} [{plays}] -> {res}")
    elif t == "raise_called":
      self.replay(f"    {e['name']} called by seat {e['seat']} (Team {e['team']}), "
                  f"seat {e['responder_seat']} to respond")
    elif t == "raise_accepted":
      self.replay(f"    accepted by seat {e['seat']}: stake {e['stake']}, raise right {e['raise_right']}")
    elif t == "raise_declined":
      self.replay(f"    declined by seat {e['seat']}: callers score {e['points']}")
    elif t == "hand_result":
      self.replay(f"    HAND RESULT: Team {e['winner_team']} +{e['points']} ({e['reason']}); "
                  f"score A {e['scores']['A']} – B {e['scores']['B']}")
    elif t == "match_end":
      self.replay()
      self.replay(f"=== MATCH OVER: Team {e['winner']} wins {e['scores']['A']}–{e['scores']['B']} "
                  f"after {e['hands_played']} hands ===")

  def close(self, engine: TrucoMatch, summary: dict[str, Any]) -> None:
    self.close_transcript()
    if not self.out_dir:
      return
    with open(os.path.join(self.out_dir, "engine_events.jsonl"), "w", encoding="utf-8") as f:
      f.write(engine.serialize_events())
    with open(os.path.join(self.out_dir, "replay.txt"), "w", encoding="utf-8") as f:
      f.write("\n".join(self._replay_lines) + "\n")
    with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
