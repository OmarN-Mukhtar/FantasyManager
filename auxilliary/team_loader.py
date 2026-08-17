"""Loads a manager's current squad from the FPL API into the session-state
shape the pitch view renders (see auxilliary/app.py)."""
import requests

BASE = "https://fantasy.premierleague.com/api"
TIMEOUT = 10
SEASON_WILDCARDS = 2  # one per half-season; ponytail: doesn't split which half is left


def last_finished_gw():
    """Most recent gameweek with results in. ponytail: in the off-season every
    event shows finished=True, so this naturally lands on the season's last GW."""
    events = requests.get(f"{BASE}/bootstrap-static/", timeout=TIMEOUT).json()["events"]
    finished = [e["id"] for e in events if e["finished"]]
    if not finished:
        raise ValueError("No finished gameweeks yet this season.")
    return max(finished)


def load_team_squad(team_id, players_df, gw=None):
    """Fetch a manager's picks for `gw` (default: last finished) and shape
    them into the pitch-view squad dict.

    players_df: data/players.csv loaded as a DataFrame (id, position, now_cost, ...).
    Returns a dict matching st.session_state.squad's schema.
    """
    gw = gw or last_finished_gw()

    picks_resp = requests.get(f"{BASE}/entry/{team_id}/event/{gw}/picks/", timeout=TIMEOUT)
    picks_resp.raise_for_status()
    picks_data = picks_resp.json()

    history_resp = requests.get(f"{BASE}/entry/{team_id}/history/", timeout=TIMEOUT)
    history_resp.raise_for_status()
    chips_used = sum(1 for c in history_resp.json().get("chips", []) if c["name"] == "wildcard")

    by_element = players_df.set_index("id")
    picks = sorted(picks_data["picks"], key=lambda p: p["position"])
    starting_ids = [p["element"] for p in picks[:11]]
    bench_ids = [p["element"] for p in picks[11:15]]
    captain_id = next(p["element"] for p in picks if p["is_captain"])
    vice_captain_id = next(p["element"] for p in picks if p["is_vice_captain"])

    formation_counts = by_element.loc[starting_ids, "position"].value_counts()
    formation = (
        int(formation_counts.get("DEF", 0)),
        int(formation_counts.get("MID", 0)),
        int(formation_counts.get("FWD", 0)),
    )

    entry_history = picks_data["entry_history"]
    return {
        "starting_ids": starting_ids,
        "bench_ids": bench_ids,
        "formation": formation,
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
        "bank": entry_history["bank"] / 10,
        "squad_value": entry_history["value"] / 10,
        # ponytail: FPL's free-transfer rollover isn't exposed directly here;
        # default to 1 and let the UI stepper override it.
        "free_transfers": 1,
        "wildcards_available": max(0, SEASON_WILDCARDS - chips_used),
        "wildcard_active": picks_data.get("active_chip") == "wildcard",
        "team_id": team_id,
        "gw": gw,
    }
