import html
import random
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# `streamlit run auxilliary/app.py` puts auxilliary/ on sys.path, not the repo
# root, so the RAG/optimizer/auxilliary package imports below need this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from RAG.langchain_rag import agent, model
from auxilliary.team_loader import CHIP_LABELS, CHIP_NAMES, load_team_squad
from optimizer.squad_optimizer import VALID_FORMATIONS, optimize_squad

st.set_page_config(page_title="Fantasy Manager", page_icon=":soccer:", layout="wide")

POS_COLORS = {"GK": "#d9a916", "DEF": "#2f6fb0", "MID": "#b5333f", "FWD": "#d17a2b"}

# App-wide tokens (turf/gold palette) shared by the pitch view and the
# players table, so both tabs read as one theme instead of drifting apart.
THEME_CSS = """
<style>
:root {
  --turf-a: #163d24; --turf-b: #1c4c2c; --bench: #101410;
  --gold: #c99a2e; --gold-ink: #3a2c05;
  --panel: #ffffff; --border: #d9d6c8;
  --text-dim: #63624f; --text-faint: #99977f;
  --gk: #d9a916; --def: #2f6fb0; --mid: #b5333f; --fwd: #d17a2b;
}
.players-table-wrap {
  overflow-x: auto; border: 1px solid var(--border); border-radius: 14px;
  background: var(--panel); padding: 4px;
}
table.players { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
table.players th {
  text-align: left; font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--text-faint); font-weight: 700; padding: 10px 12px;
  border-bottom: 1px solid var(--border); white-space: nowrap;
}
table.players td {
  padding: 9px 12px; border-bottom: 1px solid var(--border);
  white-space: nowrap; font-variant-numeric: tabular-nums;
}
table.players tr:last-child td { border-bottom: none; }
.pos-chip { display: inline-flex; align-items: center; gap: 5px; font-size: 0.7rem; font-weight: 700; }
.pos-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

.app-header {
  display: flex; align-items: center; gap: 8px;
  font-weight: 800; letter-spacing: -0.02em; font-size: 1.35rem;
  margin-bottom: 4px;
}
.app-header .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--turf-b); display: inline-block; }
.app-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 18px 20px; height: 100%;
}
.app-card h3 {
  margin: 0 0 4px; font-weight: 800; font-size: 0.95rem;
}
.app-card p { margin: 0 0 14px; color: var(--text-dim); font-size: 0.85rem; }
div[data-testid="stButton"] button[kind="primary"] {
  background: var(--gold); color: var(--gold-ink); border: none; font-weight: 700;
}
</style>
"""

PITCH_CSS = """
<style>
/* st.container(key=...) stamps a real "st-key-<name>" class on the block's
   wrapper div (unlike raw st.markdown HTML, which is inserted as an
   isolated fragment and can't wrap later widgets) — that's the hook used
   to paint an actual pitch background behind the player buttons below. */
div.st-key-pitch {
  position: relative;
  background: repeating-linear-gradient(
    180deg, var(--turf-a) 0, var(--turf-a) 56px, var(--turf-b) 56px, var(--turf-b) 112px
  );
  border-radius: 10px 10px 0 0;
  padding: 18px 4px 6px;
}
div.st-key-pitch::before {
  content: ""; position: absolute; left: 50%; top: 50%;
  width: 110px; height: 110px; border: 1.5px solid rgba(255,255,255,0.45);
  border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none;
}
.pitch-halfway {
  position: absolute; left: 10px; right: 10px; top: 50%;
  border-top: 1.5px solid rgba(255,255,255,0.45); transform: translateY(-50%); pointer-events: none;
}
div.st-key-bench { background: var(--bench); border-radius: 0 0 10px 10px; padding: 10px 4px; }
.slot-label {
  text-align: center; color: rgba(255,255,255,0.85); font-size: 0.7rem;
  font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; margin-top: 4px;
}

/* Player slots render as circular, position-ringed "shirts" instead of
   plain rectangular buttons. Streamlit wraps the marker div and the
   st.button in separate sibling "element-container" divs (not as direct
   siblings of each other), so :has() is needed to bridge from the marker
   up to its container, then across to the next container's button. */
div[data-testid="stElementContainer"]:has(.shirt-marker)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button {
  border-radius: 50% !important;
  width: 52px !important; height: 52px !important; min-width: 52px !important;
  padding: 0 !important; margin: 0 auto !important;
  display: flex !important; align-items: center; justify-content: center;
  background: var(--bench) !important; color: #fff !important;
  font-size: 0.66rem !important; font-weight: 700 !important; line-height: 1.1 !important;
  border: 2.5px solid #6b6b6b !important; white-space: normal !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.35);
}
div[data-testid="stElementContainer"]:has(.shirt-marker.gk)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { border-color: var(--gk) !important; }
div[data-testid="stElementContainer"]:has(.shirt-marker.def)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { border-color: var(--def) !important; }
div[data-testid="stElementContainer"]:has(.shirt-marker.mid)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { border-color: var(--mid) !important; }
div[data-testid="stElementContainer"]:has(.shirt-marker.fwd)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { border-color: var(--fwd) !important; }
div[data-testid="stElementContainer"]:has(.shirt-marker.empty)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button {
  background: transparent !important; border-style: dashed !important;
  color: rgba(255,255,255,0.6) !important; box-shadow: none;
}
div[data-testid="stElementContainer"]:has(.shirt-marker.captain)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { box-shadow: 0 0 0 3px var(--gold) !important; }
div[data-testid="stElementContainer"]:has(.shirt-marker.vice)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button { box-shadow: 0 0 0 3px rgba(201,154,46,0.55) !important; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


@st.cache_data
def load_players_df():
    preds = pd.read_csv("data/predictions.csv")
    meta = pd.read_csv("data/players.csv")[["id", "name", "team_id", "web_name"]]
    return preds.merge(meta, left_on="player_name", right_on="name", how="inner")


def blank_squad():
    formation = random.choice(VALID_FORMATIONS)
    return {
        "starting_ids": [None] * 11,
        "bench_ids": [None] * 4,
        "formation": formation,
        "captain_id": None,
        "vice_captain_id": None,
        "bank": 100.0,
        "free_transfers": 1,
        "chips_available": {name: 2 for name in CHIP_NAMES},
        "active_chip": None,
        "team_id": None,
    }


def slot_positions(formation):
    d, m, f = formation
    return ["GK"] + ["DEF"] * d + ["MID"] * m + ["FWD"] * f


def squad_spend(squad, players):
    ids = [i for i in squad["starting_ids"] + squad["bench_ids"] if i is not None]
    return players.set_index("id").loc[ids, "now_cost"].sum() if ids else 0.0


def used_ids(squad):
    return [i for i in squad["starting_ids"] + squad["bench_ids"] if i is not None]


def total_budget(squad, players):
    # Total money in the system: what's already spent on the squad + what's
    # left in the bank. For a scratch squad nothing's spent yet, so it's £100m.
    if squad["team_id"]:
        return squad["bank"] + squad_spend(squad, players)
    return 100.0


def apply_optimizer_result(squad, players, result):
    players_by_id = players.set_index("id")
    xi_ids = result["starting_xi_ids"]
    bench_ids = [i for i in result["squad_ids"] if i not in xi_ids]

    pos_counts = players_by_id.loc[xi_ids, "position"].value_counts()
    formation = (int(pos_counts.get("DEF", 0)), int(pos_counts.get("MID", 0)), int(pos_counts.get("FWD", 0)))

    by_pos, next_idx = {}, {}
    for pid in xi_ids:
        by_pos.setdefault(players_by_id.loc[pid, "position"], []).append(pid)

    starting_ids = []
    for pos in slot_positions(formation):
        i = next_idx.get(pos, 0)
        starting_ids.append(by_pos[pos][i])
        next_idx[pos] = i + 1

    squad["formation"] = formation
    squad["starting_ids"] = starting_ids
    squad["bench_ids"] = (bench_ids + [None] * 4)[:4]
    squad["captain_id"] = result["captain_id"]
    squad["vice_captain_id"] = result["vice_captain_id"]


def recommend_captains(squad, players):
    ids = [i for i in squad["starting_ids"] if i is not None]
    if len(ids) < 2:
        return None
    top2 = players.set_index("id").loc[ids].sort_values("predicted_next_gw_points", ascending=False).head(2)
    return list(top2[["web_name", "predicted_next_gw_points"]].itertuples(index=True, name=None))


def generate_team_summary(squad, players):
    ids = squad["starting_ids"] + squad["bench_ids"]
    sub = players.set_index("id").loc[ids][["player_name", "position", "predicted_next_5_weighted"]]
    prompt = (
        "In exactly one sentence, describe this Fantasy Premier League squad's "
        "strengths and weaknesses:\n" + sub.to_string(index=False)
    )
    return model.invoke(prompt).content


def render_landing():
    st.caption("Enter your FPL team, or start building from scratch.")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='app-card'><h3>Enter my team</h3>"
            "<p>Pull your live squad, bank and chips from the FPL API.</p></div>",
            unsafe_allow_html=True,
        )
        team_id = st.text_input("FPL Team ID", key="team_id_input")
        if st.button("Load team", type="primary") and team_id:
            try:
                players = load_players_df()
                with st.spinner("Fetching your squad..."):
                    st.session_state.squad = load_team_squad(int(team_id), players)
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't load team {team_id}: {e}")

    with col2:
        st.markdown(
            "<div class='app-card'><h3>Start from scratch</h3>"
            "<p>Build a squad with a full £100m budget and no constraints.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Build a new squad"):
            st.session_state.squad = blank_squad()
            st.rerun()


def render_slot(squad, players, bucket, idx, position_hint):
    ids = squad[bucket]
    pid = ids[idx]
    label = "+"
    marker_classes = ["shirt-marker", "empty"]
    if pid is not None:
        row = players.set_index("id").loc[pid]
        label = row["web_name"]
        pos = row["position"] if bucket == "bench_ids" else position_hint
        marker_classes = ["shirt-marker", pos.lower()]
        if pid == squad["captain_id"]:
            label += " (C)"
            marker_classes.append("captain")
        elif pid == squad["vice_captain_id"]:
            label += " (VC)"
            marker_classes.append("vice")
    st.markdown(f"<div class='{' '.join(marker_classes)}'></div>", unsafe_allow_html=True)
    if st.button(label, key=f"{bucket}_{idx}", use_container_width=True):
        st.session_state.editing_slot = (bucket, idx, position_hint)
    st.markdown(f"<div class='slot-label'>{position_hint}</div>", unsafe_allow_html=True)


def render_slot_editor(squad, players):
    if "editing_slot" not in st.session_state:
        return
    bucket, idx, position_hint = st.session_state.editing_slot

    with st.container(border=True):
        st.write(f"Assign **{position_hint}** slot")
        used = {i for i in squad["starting_ids"] + squad["bench_ids"] if i is not None}
        # ponytail: bench slots aren't position-locked to keep this simple —
        # any player can sit on the bench, matching how FPL actually allows it.
        candidates = players if bucket == "bench" else players[players["position"] == position_hint]
        candidates = candidates[~candidates["id"].isin(used - {squad[bucket][idx]})]
        candidates = candidates.sort_values("predicted_next_5_weighted", ascending=False)

        options = [None] + candidates["id"].tolist()
        current = squad[bucket][idx]
        choice = st.selectbox(
            "Player", options, index=options.index(current) if current in options else 0,
            format_func=lambda pid: "— empty —" if pid is None
            else f"{players.set_index('id').loc[pid, 'web_name']} (£{players.set_index('id').loc[pid, 'now_cost']}m)",
        )
        col_a, col_b = st.columns(2)
        if col_a.button("Save"):
            squad[bucket][idx] = choice
            if choice is None:
                if squad["captain_id"] == current:
                    squad["captain_id"] = None
                if squad["vice_captain_id"] == current:
                    squad["vice_captain_id"] = None
            st.session_state.pop("team_summary_text", None)
            del st.session_state.editing_slot
            st.rerun()
        if col_b.button("Cancel"):
            del st.session_state.editing_slot
            st.rerun()


def render_pitch():
    squad = st.session_state.squad
    players = load_players_df()
    st.markdown(PITCH_CSS, unsafe_allow_html=True)

    positions = slot_positions(squad["formation"])
    rows = [("GK", [0]), ("DEF", range(1, 1 + squad["formation"][0]))]
    rows.append(("MID", range(rows[-1][1].stop, rows[-1][1].stop + squad["formation"][1])))
    rows.append(("FWD", range(rows[-1][1].stop, rows[-1][1].stop + squad["formation"][2])))

    with st.container(key="pitch"):
        st.markdown("<div class='pitch-halfway'></div>", unsafe_allow_html=True)
        for _, idx_range in rows:
            idx_range = list(idx_range)
            cols = st.columns(len(idx_range))
            for col, idx in zip(cols, idx_range):
                with col:
                    render_slot(squad, players, "starting_ids", idx, positions[idx])

    with st.container(key="bench"):
        cols = st.columns(4)
        for col, idx in zip(cols, range(4)):
            with col:
                render_slot(squad, players, "bench_ids", idx, "SUB")

    render_slot_editor(squad, players)

    st.divider()
    # For an FPL-loaded squad, `bank` is already "leftover after the loaded squad"
    # and doesn't change as you reassign slots (that's the optimizer's job).
    # For a scratch squad there's no prior bank, so recompute from £100m - spend.
    if squad["team_id"]:
        remaining = squad["bank"]
    else:
        remaining = round(100.0 - squad_spend(squad, players), 1)

    m1, m2 = st.columns(2)
    m1.metric("Remaining cash", f"£{remaining}m")
    m2.metric("Free transfers", squad["free_transfers"])

    chip_options = [None] + CHIP_NAMES
    squad["active_chip"] = st.selectbox(
        "Active chip this GW", chip_options,
        index=chip_options.index(squad["active_chip"]),
        format_func=lambda c: "— none —" if c is None else CHIP_LABELS[c],
    )

    chip_cols = st.columns(len(CHIP_NAMES))
    for col, name in zip(chip_cols, CHIP_NAMES):
        if squad["team_id"] is None:
            squad["chips_available"][name] = col.number_input(
                CHIP_LABELS[name], 0, 2, squad["chips_available"][name], key=f"chip_{name}",
            )
        else:
            col.metric(CHIP_LABELS[name], squad["chips_available"][name])

    if squad["team_id"] is None:
        squad["free_transfers"] = st.number_input("Free transfers", 0, 5, squad["free_transfers"])


def render_players_tab():
    @st.cache_data
    def load_players():
        df = pd.read_csv('data/predictions.csv')
        sentiment = pd.read_csv('data/sentiment_analysis.csv')
        return df.merge(sentiment, on='player_name', how='left')

    players = load_players()

    col1, col2, col3, col4 = st.columns(4)
    search = col1.text_input("Search player")
    positions = col2.multiselect("Position", ['GK', 'DEF', 'MID', 'FWD'])
    team = col3.selectbox("Team", ['All'] + sorted(players['team'].dropna().unique().tolist()))
    max_price = col4.slider("Max price (£M)", 3.5, 15.5, 15.5, 0.5)

    if search:
        players = players[players['player_name'].str.contains(search, case=False, na=False)]
    if positions:
        players = players[players['position'].isin(positions)]
    if team != 'All':
        players = players[players['team'] == team]
    players = players[players['now_cost'] <= max_price]

    players = players.sort_values('predicted_next_5_weighted', ascending=False)

    rows = []
    for r in players.itertuples(index=False):
        dot_color = POS_COLORS.get(r.position, "#999")
        sentiment = "—" if pd.isna(r.sentiment_score) else f"{r.sentiment_score:.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r.player_name))}</td>"
            f"<td><span class='pos-chip'><span class='pos-dot' style='background:{dot_color}'></span>{r.position}</span></td>"
            f"<td>{html.escape(str(r.team))}</td>"
            f"<td>£{r.now_cost:.1f}</td>"
            f"<td>{r.current_season_points:.0f}</td>"
            f"<td>{sentiment}</td>"
            f"<td>{r.predicted_next_gw_points:.1f}</td>"
            f"<td>{r.predicted_next_5_weighted:.1f}</td>"
            f"<td>{html.escape(str(r.next_5_fixtures))}</td>"
            "</tr>"
        )

    st.markdown(
        "<div class='players-table-wrap'><table class='players'><thead><tr>"
        "<th>Player</th><th>Pos</th><th>Team</th><th>£M</th><th>Season pts</th>"
        "<th>Sentiment</th><th>Next GW</th><th>Next 5 (weighted)</th><th>Next 5 fixtures</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_side_panel():
    squad = st.session_state.squad
    players = load_players_df()

    st.subheader("Squad")
    if st.button("⚡ Optimize team", use_container_width=True):
        with st.spinner("Solving..."):
            current = used_ids(squad) if squad["team_id"] else None
            result = optimize_squad(
                players, budget=total_budget(squad, players),
                free_transfers=squad["free_transfers"],
                active_chip=squad["active_chip"], current_ids=current,
            )
            apply_optimizer_result(squad, players, result)
        st.session_state.pop("team_summary_text", None)
        st.success(
            f"{result['transfers_made']} transfer(s), "
            f"-{result['points_penalty']} pt hit, "
            f"{result['predicted_total']} predicted pts (next 5 GWs)."
        )
        st.rerun()

    captains = recommend_captains(squad, players)
    if captains:
        (c_id, c_name, c_pts), (v_id, v_name, v_pts) = captains
        st.caption(f"Captain pick: **{c_name}** ({c_pts:.1f} pts) · Vice: **{v_name}** ({v_pts:.1f} pts)")
        if squad["active_chip"] == "3xc":
            st.caption("Triple Captain active — captain scores 3x this GW.")
        elif squad["active_chip"] == "bboost":
            st.caption("Bench Boost active — bench players also score this GW.")

    if all(i is not None for i in squad["starting_ids"] + squad["bench_ids"]):
        if st.button("Get team summary"):
            with st.spinner("Thinking..."):
                try:
                    st.session_state.team_summary_text = generate_team_summary(squad, players)
                except Exception as e:
                    st.error(f"Couldn't generate summary: {e}")
        if st.session_state.get("team_summary_text"):
            st.info(st.session_state.team_summary_text)

    st.divider()
    st.subheader("Chat")
    render_chat()


def render_chat():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("Ask about your team, predictions, or FPL advice!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Watching the tapes..."):
                try:
                    chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ]
                    squad = st.session_state.get("squad")
                    if squad:
                        players = load_players_df()
                        ids = used_ids(squad)
                        names = players.set_index("id").loc[ids, "player_name"].tolist() if ids else []
                        context = "Current squad on screen: " + (", ".join(names) if names else "empty")
                        chat_history = [{"role": "system", "content": context}] + chat_history
                    response = agent.invoke({"messages": chat_history})["messages"][-1].content
                    if not response:
                        response = "I couldn't generate a response. Please try again."
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)

            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


st.markdown("<div class='app-header'><span class='dot'></span>Fantasy Manager</div>", unsafe_allow_html=True)

if "squad" not in st.session_state:
    st.session_state.squad = None

if st.session_state.squad is None:
    render_landing()
else:
    tab_team, tab_players = st.tabs(["🏟 Team", "📋 Players"])
    with tab_team:
        pitch_col, side_col = st.columns([3, 1])
        with pitch_col:
            render_pitch()
        with side_col:
            render_side_panel()
    with tab_players:
        render_players_tab()
