"""Squad optimizer: picks the 15-player squad (and starting XI) that maximizes
predicted points over the next 5 gameweeks, subject to budget, club, and
position-quota constraints — an integer program (0/1 knapsack with extra
constraints), solved with PuLP/CBC.

When there's an existing squad and no active chip that grants free transfers,
extra transfers cost 4 points each, and it's only worth taking a hit if the
point gain outweighs it. We handle that by solving the ILP once per candidate
transfer count K (0..15, capped by squad size) with "at most K new players"
as a constraint, then picking the K whose
(optimum - 4*max(0, K - free_transfers)) is highest.

Of the four chips (wildcard, free hit, bench boost, triple captain), only
wildcard and free hit change squad-selection mechanics — both give unlimited
free transfers for the ILP's purposes (free hit's squad reverting next GW
isn't modeled since this optimizer only reasons about the current week).
Bench boost and triple captain don't change which 15 players to pick; they
just change how those players score, so they're informational only here.
"""
import pulp
import pandas as pd

POSITION_QUOTAS = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
SQUAD_SIZE = 15
CLUB_LIMIT = 3
POINTS_PER_HIT = 4
NO_PENALTY_CHIPS = {'wildcard', 'freehit'}

# formation -> (DEF, MID, FWD), GK is always 1
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3),
    (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1),
]


def _solve_squad(players, budget, max_new, current_ids):
    """One ILP solve: best squad of 15 within budget/quotas/club-limit,
    using at most `max_new` players not in `current_ids`. Returns
    (selected_ids, objective_value) or (None, None) if infeasible."""
    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    x = {p['id']: pulp.LpVariable(f"x_{p['id']}", cat="Binary") for p in players}

    prob += pulp.lpSum(x[p['id']] * p['predicted_next_5_weighted'] for p in players)

    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    prob += pulp.lpSum(x[p['id']] * p['now_cost'] for p in players) <= budget

    for pos, quota in POSITION_QUOTAS.items():
        prob += pulp.lpSum(x[p['id']] for p in players if p['position'] == pos) == quota

    for club in {p['team_id'] for p in players}:
        prob += pulp.lpSum(x[p['id']] for p in players if p['team_id'] == club) <= CLUB_LIMIT

    if current_ids is not None:
        new_players = [p for p in players if p['id'] not in current_ids]
        prob += pulp.lpSum(x[p['id']] for p in new_players) <= max_new

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None

    selected = [p['id'] for p in players if x[p['id']].value() > 0.5]
    return selected, pulp.value(prob.objective)


def _best_starting_xi(squad_players):
    """Pick captain-worthy starting XI: try every valid formation, take the
    top N players per position by next-GW points, keep the best-scoring one.
    Small enough (13 outfield candidates, 7 formations) that brute force
    beats a second ILP."""
    by_pos = {
        pos: sorted(
            [p for p in squad_players if p['position'] == pos],
            key=lambda p: p['predicted_next_gw_points'], reverse=True,
        )
        for pos in POSITION_QUOTAS
    }

    best_xi, best_score = None, -1
    for defn, mid, fwd in VALID_FORMATIONS:
        counts = {'GK': 1, 'DEF': defn, 'MID': mid, 'FWD': fwd}
        if any(len(by_pos[pos]) < n for pos, n in counts.items()):
            continue
        xi = [p for pos, n in counts.items() for p in by_pos[pos][:n]]
        score = sum(p['predicted_next_gw_points'] for p in xi)
        if score > best_score:
            best_xi, best_score = xi, score

    return best_xi


def optimize_squad(players_df, budget, free_transfers=1, active_chip=None, current_ids=None):
    """Optimize a 15-player squad.

    players_df: DataFrame with columns id, position, team_id, now_cost,
        predicted_next_5_weighted, predicted_next_gw_points.
    budget: total money available (existing squad's now_cost sum + bank, or
        100.0 for a from-scratch squad).
    active_chip: one of 'wildcard', 'freehit', 'bboost', '3xc', or None.
        Only wildcard/freehit affect selection (unlimited free transfers).
    current_ids: ids of the currently-owned squad, or None/empty for scratch.
    Returns a dict: squad_ids, starting_xi_ids, captain_id, vice_captain_id,
        transfers_made, points_penalty, predicted_total.
    """
    players = players_df.to_dict('records')
    current_ids = set(current_ids or [])

    if active_chip in NO_PENALTY_CHIPS or not current_ids:
        # No transfer-cost tradeoff to weigh: either hits are free (wildcard/
        # free hit) or there's no prior squad to compare against (scratch build).
        selected, objective = _solve_squad(players, budget, SQUAD_SIZE, current_ids or None)
        if selected is None:
            raise ValueError("No feasible squad found within budget/constraints.")
        transfers_made = len(set(selected) - current_ids)
        penalty = 0
    else:
        best = None  # (net_score, K, selected, objective)
        for k in range(0, SQUAD_SIZE + 1):
            selected, objective = _solve_squad(players, budget, k, current_ids)
            if selected is None:
                continue
            hits = max(0, k - free_transfers)
            net = objective - POINTS_PER_HIT * hits
            if best is None or net > best[0]:
                best = (net, k, selected, objective)
        if best is None:
            raise ValueError("No feasible squad found within budget/constraints.")
        _, k, selected, objective = best
        transfers_made = k
        penalty = POINTS_PER_HIT * max(0, k - free_transfers)

    squad_players = [p for p in players if p['id'] in selected]
    xi = _best_starting_xi(squad_players)
    xi_sorted = sorted(xi, key=lambda p: p['predicted_next_gw_points'], reverse=True)

    return {
        'squad_ids': selected,
        'starting_xi_ids': [p['id'] for p in xi],
        'captain_id': xi_sorted[0]['id'],
        'vice_captain_id': xi_sorted[1]['id'],
        'transfers_made': transfers_made,
        'points_penalty': penalty,
        'predicted_total': round(objective - penalty, 2),
    }


def _demo():
    """ponytail: smallest runnable check — a synthetic 30-player pool covering
    every position/club, verifying quotas, budget, and the transfer-penalty
    tradeoff all hold."""
    import itertools
    rows = []
    pid = itertools.count(1)
    for pos, n in [('GK', 6), ('DEF', 10), ('MID', 10), ('FWD', 6)]:
        for i in range(n):
            rows.append({
                'id': next(pid), 'position': pos, 'team_id': i % 5,
                'now_cost': 4.0 + (i % 4), 'predicted_next_5_weighted': 10.0 + i,
                'predicted_next_gw_points': 2.0 + i * 0.3,
            })
    df = pd.DataFrame(rows)

    # Scratch build: must respect quotas and budget.
    result = optimize_squad(df, budget=100.0)
    assert len(result['squad_ids']) == SQUAD_SIZE
    squad = df[df['id'].isin(result['squad_ids'])]
    assert (squad['now_cost'].sum()) <= 100.0 + 1e-6
    for pos, quota in POSITION_QUOTAS.items():
        assert (squad['position'] == pos).sum() == quota
    assert len(result['starting_xi_ids']) == 11
    assert result['captain_id'] != result['vice_captain_id']

    # Existing squad, no wildcard: 0 transfers should be at least as good an
    # option as a full rebuild once hits are priced in, i.e. penalty logic ran.
    current = result['squad_ids']
    result2 = optimize_squad(df, budget=100.0, free_transfers=1, current_ids=current)
    assert result2['points_penalty'] == POINTS_PER_HIT * max(0, result2['transfers_made'] - 1)

    # Wildcard / free hit: no penalty regardless of turnover.
    result3 = optimize_squad(df, budget=100.0, active_chip='wildcard', current_ids=current)
    assert result3['points_penalty'] == 0
    result4 = optimize_squad(df, budget=100.0, active_chip='freehit', current_ids=current)
    assert result4['points_penalty'] == 0

    # Bench boost / triple captain don't grant free transfers.
    result5 = optimize_squad(df, budget=100.0, active_chip='bboost', free_transfers=1, current_ids=current)
    assert result5['points_penalty'] == POINTS_PER_HIT * max(0, result5['transfers_made'] - 1)

    print("optimizer self-check passed")


if __name__ == "__main__":
    _demo()
