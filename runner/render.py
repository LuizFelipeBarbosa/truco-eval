"""Render a seat's observation dict to text for prompts and replays.

Only fields present in ``TrucoMatch.observation(seat)`` are used, so the
rendered text can never contain more than that seat is entitled to see.
"""

from __future__ import annotations

TALK_LINES_FROM_EARLIER_HANDS = 10


def _team_label(team: str | None) -> str:
  return "parda" if team is None else f"Team {team}"


def render_observation(obs: dict) -> str:
  n = obs["num_players"]
  seat, team = obs["seat"], obs["team"]
  partners = obs["partner_seats"]
  opponents = [s for s in range(n) if s % 2 != seat % 2]
  scores = obs["scores"]
  lines = [
      "=== Truco match state ===",
      f"Score: Team A {scores['A']} – Team B {scores['B']} (first to {obs['target']} wins).",
      f"Hand #{obs['hand_index'] + 1}. Dealer: seat {obs['dealer']}. Mão (leads trick 1): seat {obs['mao']}.",
      f"You are seat {seat} (Team {team}). "
      + (f"Partner: seat {', '.join(map(str, partners))}. " if partners else "You have no partner. ")
      + f"Opponents: seats {', '.join(map(str, opponents))}.",
  ]
  ht = obs["hand_type"]
  if ht == "mao_de_onze":
    lines.append("Hand type: MÃO DE ONZE (fixed stake 3, no raises allowed).")
  elif ht == "mao_de_ferro":
    lines.append("Hand type: MÃO DE FERRO (blind hand, fixed stake 1, no raises; the winner wins the match).")
  else:
    lines.append("Hand type: normal.")
  lines.append(
      f"Vira: {obs['vira']}. Manilha rank: {obs['manilha_rank']}. "
      f"Strength order this hand: {obs['strength_order']}."
  )
  if obs.get("hand") is not None:
    lines.append(f"Your hand: {', '.join(obs['hand']) if obs['hand'] else '(no cards left)'}.")
  elif obs.get("blind_positions") is not None:
    lines.append(
        "Your cards are face down; you cannot see them. Remaining positions: "
        f"{', '.join(map(str, obs['blind_positions']))}."
    )
  if obs.get("partner_hand"):
    for s, cards in obs["partner_hand"].items():
      lines.append(f"Partner seat {s}'s hand (visible for the mão de onze decision): {', '.join(cards)}.")
  if obs.get("partner_hand_at_deal"):
    for s, cards in obs["partner_hand_at_deal"].items():
      lines.append(f"Partner seat {s}'s cards as dealt (seen during the mão de onze decision): {', '.join(cards)}.")

  call = obs.get("pending_call")
  stake_line = f"Stake: {obs['stake']}."
  if call:
    stake_line += (
        f" PENDING CALL: {call['name']} (to {call['value']}) by seat {call['caller_seat']} "
        f"(Team {call['calling_team']}); seat {call['responder_seat']} must respond."
    )
  else:
    stake_line += " No pending call."
  if obs["raises_allowed"]:
    rr = obs["raise_right"]
    stake_line += " Right to raise: " + (", ".join(f"Team {t}" for t in rr) if rr else "nobody") + "."
  else:
    stake_line += " Raises are not allowed in this hand."
  lines.append(stake_line)

  lines.append("Tricks this hand:")
  any_trick = False
  for tr in obs["trick_results"]:
    any_trick = True
    plays = ", ".join(f"seat {p['seat']} {p['card']}" for p in tr["plays"])
    if tr["winner_team"] is None:
      result = "PARDA (draw)"
    else:
      result = f"{_team_label(tr['winner_team'])} wins (seat {tr['winner_seat']})"
    lines.append(f"  Trick {tr['trick'] + 1} (leader seat {tr['leader']}): {plays} → {result}")
  if not obs["terminal"] and obs["phase"] != "MAO_DE_ONZE":
    cur = obs["current_trick"]
    played = ", ".join(f"seat {p['seat']} {p['card']}" for p in cur) if cur else "no cards yet"
    lines.append(
        f"  Trick {obs['trick_index'] + 1} (in progress, leader seat {obs['trick_leader']}): {played}"
    )
    any_trick = True
  if not any_trick:
    lines.append("  (none yet)")

  hh = obs["hand_history"]
  if hh:
    parts = []
    for h in hh:
      how = h["reason"]
      if how == "tricks":
        how = "tricks " + "/".join("parda" if t is None else t for t in h["tricks"])
      elif how in ("fold", "decline"):
        how = f"{how} by seat {h['by_seat']}"
      parts.append(f"#{h['hand'] + 1} Team {h['winner_team']} +{h['points']} ({how})")
    lines.append("Previous hands: " + "; ".join(parts) + ".")
  else:
    lines.append("Previous hands: none.")

  talk = obs["talk"]
  current = [t for t in talk if t["hand"] == obs["hand_index"]]
  earlier = [t for t in talk if t["hand"] != obs["hand_index"]][-TALK_LINES_FROM_EARLIER_HANDS:]
  if current or earlier:
    lines.append("Table talk (public, unverified):")
    for t in earlier + current:
      lines.append(f"  [hand #{t['hand'] + 1}, trick {t['trick'] + 1}] seat {t['seat']}: \"{t['text']}\"")
  else:
    lines.append("Table talk: none so far.")
  lines.append(f"Seat to act: {obs['to_act']}.")
  return "\n".join(lines)
