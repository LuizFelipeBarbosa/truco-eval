"""Prompt templates for Truco decisions, in the harness's f-string style.

Every template ends with a ``{rethink_prompt}`` slot (empty on the first
attempt) exactly like ``prompt_templates.*_RETHINK_APPENDED`` in the harness.
"""

from runner.rules_text import RULES_FOR_MODELS

SYSTEM_INSTRUCTION = (
    "You are playing the card game Truco Paulista in a benchmark. You control one seat. "
    "The complete rules follow; they are the only rules in force.\n\n"
    + RULES_FOR_MODELS
    + "\n\nResponse format (mandatory): reason briefly, then finish with these lines:\n"
    "Talk: <optional public message to the table, at most 200 characters; omit the line to say nothing>\n"
    "Final Answer: <exactly one legal action, copied verbatim from the legal action list>\n"
    "The Final Answer line must be the last line of your reply. Talk never counts as an action."
)

DECISION_TYPE_CARD_PLAY = "card_play"
DECISION_TYPE_RAISE_RESPONSE = "raise_response"
DECISION_TYPE_MAO_DE_ONZE = "mao_de_onze"
DECISION_TYPE_MAO_DE_FERRO = "mao_de_ferro"

_ANSWER_FORMAT = """Reason step by step, then end your reply with:
Talk: <optional message to the table, at most 200 characters, or omit this line>
Final Answer: <one action copied exactly from the legal action list>
{rethink_prompt}"""

CARD_PLAY = """{readable_state_str}

It is your turn, seat {seat} (Team {team}). You may play a card{raise_clause}, or fold.
The legal actions are exactly:
{legal_actions}

""" + _ANSWER_FORMAT

RAISE_RESPONSE = """{readable_state_str}

Seat {caller_seat} (Team {calling_team}) has called {call_name} (stake would become {call_value}). \
You, seat {seat} (Team {team}), must respond before any card is played: ACCEPT (stake becomes {call_value} and your team gains the right to raise), \
DECLINE (hand ends now, Team {calling_team} scores {pre_call_stake}){raise_back_clause}.
The legal actions are exactly:
{legal_actions}

""" + _ANSWER_FORMAT

MAO_DE_ONZE = """{readable_state_str}

MÃO DE ONZE: your team has 11 points. You, seat {seat}, decide for Team {team}. \
Both partners' cards are shown above. MAO_PLAY plays this hand for a fixed stake of 3 with no raises allowed (folding stays allowed); \
MAO_FORFEIT gives the opponents 1 point and deals the next hand.
The legal actions are exactly:
{legal_actions}

""" + _ANSWER_FORMAT

MAO_DE_FERRO = """{readable_state_str}

MÃO DE FERRO: both teams have 11 points. This hand is played blind for a stake of 1, no raises; the winner of the hand wins the match. \
You cannot see your cards. Choose a face-down position to play (each card is revealed when played), or fold.
The legal actions are exactly:
{legal_actions}

""" + _ANSWER_FORMAT

RETHINK = """
Your previous reply did not contain exactly one legal action, so it was rejected. Your previous reply was:
<<<
{generation}
>>>
Answer again now. The legal actions are exactly:
{legal_actions}
End with "Final Answer: X" where X is copied exactly from that list. If this reply is also illegal, a uniformly random legal action will be played for you."""

TEMPLATE_BY_DECISION_TYPE = {
    DECISION_TYPE_CARD_PLAY: CARD_PLAY,
    DECISION_TYPE_RAISE_RESPONSE: RAISE_RESPONSE,
    DECISION_TYPE_MAO_DE_ONZE: MAO_DE_ONZE,
    DECISION_TYPE_MAO_DE_FERRO: MAO_DE_FERRO,
}


def decision_type_for(observation: dict) -> str:
  phase = observation["phase"]
  if phase == "MAO_DE_ONZE":
    return DECISION_TYPE_MAO_DE_ONZE
  if phase == "RAISE_RESPONSE":
    return DECISION_TYPE_RAISE_RESPONSE
  if observation["hand_type"] == "mao_de_ferro":
    return DECISION_TYPE_MAO_DE_FERRO
  return DECISION_TYPE_CARD_PLAY


def format_legal_actions(legal_actions) -> str:
  return "\n".join(f"- {a}" for a in legal_actions)


def prompt_substitutions(observation: dict, legal_actions, readable_state_str: str) -> dict:
  """Substitutions for the template chosen by ``decision_type_for``."""
  subs = {
      "readable_state_str": readable_state_str,
      "seat": observation["seat"],
      "team": observation["team"],
      "legal_actions": format_legal_actions(legal_actions),
  }
  raise_names = [a for a in legal_actions if a in ("TRUCO", "SEIS", "NOVE", "DOZE")]
  subs["raise_clause"] = (
      f", call {raise_names[0]} (raise the stake to "
      f"{ {'TRUCO': 3, 'SEIS': 6, 'NOVE': 9, 'DOZE': 12}[raise_names[0]] })"
      if raise_names else ""
  )
  call = observation.get("pending_call")
  if call:
    subs.update({
        "caller_seat": call["caller_seat"],
        "calling_team": call["calling_team"],
        "call_name": call["name"],
        "call_value": call["value"],
        "pre_call_stake": call["pre_call_stake"],
        "raise_back_clause": (
            ", or RAISE (accept and immediately raise to the next step)"
            if "RAISE" in legal_actions else ""
        ),
    })
  return subs
