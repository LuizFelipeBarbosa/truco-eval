"""The rules document given to every model as its system prompt (minus §13)."""

RULES_FOR_MODELS = """Truco Paulista — Benchmark Rules

This is the complete rule set used by the benchmark. It contains no strategy advice.

Parameters fixed for this benchmark: 40-card deck, target 12, ladder 1-3-6-9-12, no envido, no flor, mão de onze as in §9, free-text table talk as in §11.

1. Players and teams
- The game is played by N ∈ {2, 4, 6} players in seats numbered 0 … N−1. The benchmark default is N = 4.
- Seats alternate teams: even seats are Team A, odd seats are Team B. With N = 4, seats 0 and 2 are partners, seats 1 and 3 are partners. With N = 2 each player is their own team.
- Play proceeds in seat order: after seat i, seat (i + 1) mod N acts.
- Partners share a score. A team wins or loses together.

2. The deck
40 cards: ranks 4, 5, 6, 7, Q, J, K, A, 2, 3 in each of the four suits ♣ ♥ ♠ ♦. There are no 8s, 9s, 10s, or jokers.

3. The deal
- One player is the dealer. The dealer rotates by one seat after every hand. Seat N−1 deals the first hand.
- The player after the dealer is the mão (first player). The mão plays first in trick 1.
- Each player receives 3 cards, face down, seen only by that player.
- One further card is turned face up: the vira. It is visible to everyone and belongs to no one.
- The remaining cards are not used in this hand.

4. Card strength
4.1 Manilhas
The cyclic rank order is: 4 → 5 → 6 → 7 → Q → J → K → A → 2 → 3 → (back to 4).
The rank immediately after the vira's rank in this cycle is the manilha rank. All four cards of that rank are manilhas, the strongest cards of the hand.
Vira rank → Manilha rank: 4→5, 5→6, 6→7, 7→Q, Q→J, J→K, K→A, A→2, 2→3, 3→4.
Manilhas are ranked among themselves by suit: ♣ (clubs) > ♥ (hearts) > ♠ (spades) > ♦ (diamonds). The club manilha is the single strongest card of the hand.
4.2 All other cards
Below the manilhas, cards rank by rank only — suit does not matter and equal ranks tie:
3 > 2 > A > K > J > Q > 7 > 6 > 5 > 4
Note that J beats Q. The manilha rank is removed from this list for the hand (e.g., if the vira is K, every A is a manilha and no A appears in the ordinary list).
4.3 Full order, example
Vira = 7. Manilha rank = Q. Strength from highest to lowest:
Q♣ > Q♥ > Q♠ > Q♦ > 3 > 2 > A > K > J > 7 > 6 > 5 > 4
Here the four 3s tie with each other, the four 2s tie with each other, and so on.

5. Playing a trick
- A hand consists of up to 3 tricks.
- In each trick every player plays one card face up, in seat order, starting with the trick's leader. There is no obligation to follow suit or to beat the previous card: any card in hand may be played.
- The trick is won by the team that played the strongest card.
- If the strongest card is tied — the same rank played by players on both teams, and no stronger card was played — the trick is a parda (draw). Ties between two players on the same team are not a parda; that team simply wins the trick.
- The leader of trick 1 is the mão. The leader of each later trick is the player who played the winning card of the previous trick. After a parda, the leader of the previous trick leads again.

6. Winning the hand
Let r₁, r₂, r₃ be the results of the three tricks (Team A, Team B, or parda).
- A team that wins two tricks wins the hand. If the same team wins tricks 1 and 2, the hand ends immediately; trick 3 is not played.
- If trick 1 is a parda, the hand is won by whichever team wins the first later trick that is not a parda, and the hand ends at that trick. If all three tricks are pardas, the mão's team wins the hand.
- If trick 1 has a winner and trick 2 is a parda, the winner of trick 1 wins the hand immediately; trick 3 is not played.
- If tricks 1 and 2 are won by different teams, trick 3 decides. If trick 3 is a parda, the winner of trick 1 wins the hand.

7. Stakes and Truco
7.1 The stake
Every hand starts with a stake of 1 point. The stake can be raised along the fixed ladder: 1 → 3 → 6 → 9 → 12
The raises are called, in order, Truco (to 3), Seis (to 6), Nove (to 9), and Doze (to 12). Nothing above 12 exists.
7.2 Calling a raise
- A player may call a raise only on their own turn, before playing a card, and only if their team currently holds the right to raise (§7.4).
- A raise may be called in any trick, including trick 3, and at any point in the hand where the player is to act.
- When a raise is called, play stops and the other team must respond before any card is played.
7.3 Responding to a raise
The responding team's player who is next to act in seat order gives one of three responses:
- Accept: the stake becomes the raised amount. The caller then plays their card (or acts) as normal. The right to raise passes to the accepting team.
- Decline: the hand ends immediately. The calling team scores the stake as it was before the call (1 if Truco was declined, 3 if Seis was declined, 6 if Nove was declined, 9 if Doze was declined).
- Raise back: this counts as accepting and immediately calling the next step of the ladder (e.g., answering Truco with Seis). The original calling team must now respond to the new raise in the same way. Raising back is not possible when the current call is Doze.
7.4 The right to raise
- At the start of the hand both teams hold the right to raise.
- Whenever a raise is accepted (including by raising back), the right to raise belongs only to the team that accepted. The team whose raise was accepted cannot raise again until the opponents have raised and they have accepted in turn.
- At a stake of 12 no team may raise.
7.5 What the stake is worth
At the end of the hand, the winning team scores the current stake.

8. Folding
A player may fold on their own turn (including when responding to a raise, which counts as declining). Folding ends the hand and the opposing team scores the current stake. Partners cannot veto a fold.

9. Mão de onze (hand of eleven)
Checked at the start of every hand, after the deal:
- Exactly one team has 11 points. Before any card is played, the players of that team see each other's cards (this is the only time partners' cards are ever visible). That team's mão-side player then chooses:
  - Play: the hand is played for a fixed stake of 3. No raises may be called by either team; folding remains allowed.
  - Forfeit: the hand is not played and the opposing team scores 1 point.
- Both teams have 11 points (mão de ferro). The hand is played for a fixed stake of 1, no raises allowed, and no player sees their own cards: each player's three cards are held face down in positions 1, 2, 3 and are played by position. Each card is revealed when played. The hand is otherwise resolved normally. The winner of the hand wins the match.

10. Winning the match
Points accumulate across hands. The first team to reach 12 or more points at the end of a hand wins the match. Points never carry over past 12; the match simply ends.

11. Table talk
- On any turn, a player may say something to the table, at most 200 characters, together with their action or response.
- Everything said is heard by all players on both teams. There is no private channel.
- Talk has no effect on the rules: it cannot call, accept, decline, or fold. Only the explicit action does.
- Players may say anything, including false statements about their cards. Nothing said is verified.

12. What each player knows
At every decision a player knows:
- their own hand (except in mão de ferro);
- their partner's hand only during a mão de onze decision;
- the vira and therefore the manilha rank;
- every card played so far this hand and who played it;
- the result of every completed trick;
- the current stake, any pending call, which team holds the right to raise;
- both teams' scores, who dealt, who is mão, and whose turn it is;
- all table talk so far;
- the list of actions legal for them right now.
Players never see the unused cards, the opponents' hands, or (outside mão de onze) their partner's hand.

14. Worked examples
Example A — manilhas and a parda. Vira = A, so 2s are manilhas (and A is an ordinary card this hand). Trick 1: seat 0 plays 3♠, seat 1 plays 3♦, seat 2 plays K♣, seat 3 plays 7♥. The strongest cards are the two 3s, played by seat 0 (Team A) and seat 1 (Team B): parda. Seat 0 leads trick 2 again. Trick 2: seat 0 plays 2♦ (manilha), seat 1 plays 2♣ (manilha, clubs), seat 2 plays A♥ (ordinary A), seat 3 plays 4♣. Strongest card: 2♣ by seat 1 → Team B wins trick 2 and, because trick 1 was a parda, Team B wins the hand. Trick 3 is not played.
Example B — declined Truco. Stake 1. On seat 2's turn in trick 1, seat 2 calls Truco. Seat 3 is the next opponent to act and answers Decline. The hand ends; Team A scores 1 (the stake before the call).
Example C — raise back. Stake 1. Seat 1 calls Truco. Seat 2 answers Seis (raise back). Seat 3 (Team B) must respond: Accept. Stake is now 6, the right to raise belongs to Team B, and seat 1 plays their card. Later in the hand seat 3 may call Nove; seat 0 or 2 may not.
Example D — tricks split, third decides. Team A wins trick 1, Team B wins trick 2, trick 3 is a parda. Team A wins the hand (winner of trick 1).
Example E — mão de onze. Team A has 11, Team B has 6. After the deal, seats 0 and 2 see each other's cards. Seat 0 chooses Play: the hand is worth 3, nobody may raise. If Team A wins the hand they reach 14 → match over, Team A wins. If seat 0 had chosen Forfeit, Team B would score 1 (now 7) and the next hand would be dealt.

15. Quick reference
- Deck: 40 cards: 4 5 6 7 Q J K A 2 3 × ♣ ♥ ♠ ♦
- Cards per player: 3
- Manilha: rank after the vira in 4→5→6→7→Q→J→K→A→2→3→4; suits ♣ > ♥ > ♠ > ♦
- Other cards: 3 > 2 > A > K > J > Q > 7 > 6 > 5 > 4; suits irrelevant; equal ranks tie
- Follow suit: never required
- Hand: best of 3 tricks; parda rules in §6
- Ladder: 1 → 3 (Truco) → 6 (Seis) → 9 (Nove) → 12 (Doze)
- Decline: callers score the pre-call stake
- Raise right: both teams at start; after an accept, only the accepting team
- Fold: opponents score current stake
- Mão de onze: one team at 11: see partner's cards, play for 3 or forfeit 1; both at 11: blind hand for 1
- Match: first to 12
- Talk: free text, public, unverified, no rule effect
"""
