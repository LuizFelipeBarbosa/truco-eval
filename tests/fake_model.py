"""A fake harness model that answers prompts by reading the legal list."""

from __future__ import annotations

import random
import re

from game_arena.harness import model_generation

_LEGAL_BLOCK = re.compile(r"The legal actions are exactly:\n((?:- .+\n?)+)")


def legal_from_prompt(prompt: str) -> list[str]:
  m = list(_LEGAL_BLOCK.finditer(prompt))
  assert m, "prompt must show the legal list"
  return [line[2:].strip() for line in m[-1].group(1).strip().split("\n")]


class FakeModel:
  """Picks a random legal action from the prompt; sometimes answers illegally."""

  def __init__(self, seed: int = 0, illegal_prob: float = 0.0, talk_prob: float = 0.3,
               responses: list[str] | None = None):
    self._rng = random.Random(seed)
    self._illegal_prob = illegal_prob
    self._talk_prob = talk_prob
    self._responses = list(responses or [])
    self.calls: list[model_generation.ModelTextInput] = []
    self._model_name = "fake/model"

  @property
  def model_name(self):
    return self._model_name

  def generate_with_text_input(self, model_input):
    self.calls.append(model_input)
    if self._responses:
      text = self._responses.pop(0)
    else:
      legal = legal_from_prompt(model_input.prompt_text)
      if self._rng.random() < self._illegal_prob:
        from truco import cards as C
        held = {a[5:] for a in legal if a.startswith("PLAY ")}
        card = self._rng.choice([c for c in C.make_deck() if c not in held])
        text = f"I think I'll play the {card}, they surely hold 3♣ and 2♥.\nFinal Answer: PLAY {card}"
      else:
        talk = ("Talk: watch out, I'm strong!\n" if self._rng.random() < self._talk_prob else "")
        text = f"Let me think.\n{talk}Final Answer: {self._rng.choice(legal)}"
    return model_generation.GenerateReturn(
        main_response=text, main_response_and_thoughts=text,
        request_for_logging={"model": self._model_name},
        response_for_logging={"provider": "Fake", "usage": {"cost": 0.001}},
        prompt_tokens=100, generation_tokens=20, total_tokens=120, reasoning_tokens=0,
        duration_success_only_secs=0.0,
    )
