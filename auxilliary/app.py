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

PITCH_CSS = """
<style>
.pitch-row { background: linear-gradient(180deg, #1f8a3d, #2ba84a);
             border-top: 1px solid rgba(255,255,255,0.25);
             padding: 10px 4px; }
.pitch-row:first-child { border-radius: 10px 10px 0 0; border-top: none; }
.bench-row { background: #2c2c2c; border-radius: 0 0 10px 10px; padding: 10px 4px; }
.slot-label { text-align: center; color: white; font-size: 0.75rem; margin-top: -6px; }
</style>
"""


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
    st.title("Fantasy Manager :soccer:")
    st.caption("Enter your FPL team, or start building from scratch.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter my team")
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
        st.subheader("Start from scratch")
        if st.button("Build a new squad"):
            st.session_state.squad = blank_squad()
            st.rerun()


def render_slot(squad, players, bucket, idx, position_hint):
    ids = squad[bucket]
    pid = ids[idx]
    label = "+ Empty"
    if pid is not None:
        row = players.set_index("id").loc[pid]
        label = row["web_name"]
        if pid == squad["captain_id"]:
            label += " (C)"
        elif pid == squad["vice_captain_id"]:
            label += " (VC)"
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

    for _, idx_range in rows:
        idx_range = list(idx_range)
        st.markdown("<div class='pitch-row'>", unsafe_allow_html=True)
        cols = st.columns(len(idx_range))
        for col, idx in zip(cols, idx_range):
            with col:
                render_slot(squad, players, "starting_ids", idx, positions[idx])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='bench-row'>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, idx in zip(cols, range(4)):
        with col:
            render_slot(squad, players, "bench_ids", idx, "SUB")
    st.markdown("</div>", unsafe_allow_html=True)

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

    st.dataframe(
        players.sort_values('predicted_next_5_weighted', ascending=False)[[
            'player_name', 'position', 'team', 'now_cost', 'current_season_points',
            'sentiment_score', 'predicted_next_gw_points', 'predicted_next_5_weighted',
            'next_5_fixtures',
        ]].rename(columns={
            'player_name': 'Player', 'position': 'Pos', 'team': 'Team',
            'now_cost': '£M', 'current_season_points': 'Season pts',
            'sentiment_score': 'Sentiment', 'predicted_next_gw_points': 'Next GW',
            'predicted_next_5_weighted': 'Next 5 (weighted)',
            'next_5_fixtures': 'Next 5 fixtures',
        }),
        use_container_width=True,
        hide_index=True,
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
