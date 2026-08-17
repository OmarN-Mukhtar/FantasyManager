import html
import random
import re
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

st.set_page_config(page_title="Fantasy Manager", layout="wide")

POS_COLORS = {"GK": "#d9a916", "DEF": "#2f6fb0", "MID": "#b5333f", "FWD": "#d17a2b"}


def player_initials(name):
    parts = [p for p in re.split(r"[^A-Za-z]+", name) if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    if parts:
        return parts[0][:2].upper()
    return "?"

# App-wide tokens (turf/gold palette) shared by the pitch view and the
# players table, so both tabs read as one theme instead of drifting apart.
THEME_CSS = """
<style>
:root {
  --bg: #f2f1eb; --bg-alt: #e7e5db;
  --turf-a: #163d24; --turf-b: #1c4c2c; --bench: #101410; --line: rgba(255,255,255,0.45);
  --gold: #c99a2e; --gold-ink: #3a2c05;
  --panel: #ffffff; --border: #d9d6c8;
  --text-dim: #63624f; --text-faint: #99977f;
  --gk: #d9a916; --def: #2f6fb0; --mid: #b5333f; --fwd: #d17a2b;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1280px; margin: 0 auto; }
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
  padding: 16px 16px 18px; height: 100%;
}
.app-card h3 {
  margin: 0 0 4px; font-weight: 800; font-size: 0.95rem;
}
.app-card p { margin: 0 0 14px; color: var(--text-dim); font-size: 0.85rem; }

/* Minimal, consistent chrome for every button / input / select across the app. */
div[data-testid="stButton"] button {
  border-radius: 8px; border: 1px solid var(--border); background: #fff;
  box-shadow: none; font-weight: 600; color: #2b2a20; transition: border-color 0.15s, background 0.15s;
}
div[data-testid="stButton"] button:hover { border-color: #b3ac8f; background: #faf9f4; color: #2b2a20; }
div[data-testid="stButton"] button[kind="primary"] {
  background: var(--gold); color: var(--gold-ink); border: 1px solid var(--gold); font-weight: 700;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #b78a25; border-color: #b78a25; }

div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-testid="stNumberInput"] div {
  border-radius: 8px !important; box-shadow: none !important;
}
div[data-baseweb="select"] > div { border-color: var(--border) !important; }
div[data-baseweb="input"], div[data-baseweb="input"] input, div[data-baseweb="select"] > div {
  background: var(--panel) !important; border-color: var(--border) !important;
}
div[data-baseweb="base-input"] { background: var(--panel) !important; }

/* Stats toolbar: a slim card sitting above the pitch, echoing the
   reference mock's "Budget" card but laid out as one horizontal strip. */
div.st-key-toolbar_card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 18px 16px; margin-bottom: 14px;
}
div.st-key-toolbar_card div[data-testid="stHorizontalBlock"] { align-items: flex-start; gap: 4px; }
div.st-key-toolbar_card .stat-label {
  font-size: 0.66rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--text-faint); font-weight: 700; margin-bottom: 6px;
}
div.st-key-toolbar_card .stat-value {
  font-size: 1.1rem; font-weight: 800; color: var(--gold); font-variant-numeric: tabular-nums;
  padding-top: 6px;
}

/* Pitch card: one bordered, rounded shell holding the turf and the black
   bench strip; the title/formation/Optimize row floats directly over the
   turf instead of sitting in its own bar, corners unified via overflow. */
div.st-key-pitch_card {
  position: relative;
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  overflow: hidden;
}
div.st-key-pitch_head {
  position: absolute; top: 0; left: 0; right: 0; z-index: 2;
  padding: 14px 18px; background: transparent !important;
}
div.st-key-pitch_head div { background: transparent; }
div.st-key-pitch_head div[data-testid="stHorizontalBlock"] { align-items: center; }
.pitch-title {
  display: flex; align-items: center; gap: 10px; margin-top: -12px;
  font-weight: 800; font-size: 1rem; letter-spacing: -0.01em; color: #fff;
  text-shadow: 0 1px 3px rgba(0,0,0,0.5);
}
.formation-tag {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; color: #fff;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35); border-radius: 999px;
  padding: 3px 10px; font-variant-numeric: tabular-nums;
}
div.st-key-optimize_wrap { display: flex; align-items: flex-end; }
div.st-key-optimize_wrap div[data-testid="stButton"] button {
  border: 1px solid rgba(255,255,255,0.4); border-radius: 999px; background: rgba(255,255,255,0.14);
  padding: 5px 15px; font-size: 0.78rem; font-weight: 700; color: #fff;
  min-height: 0; width: auto; white-space: nowrap; backdrop-filter: blur(2px);
  display: block; margin-left: auto;
}
div.st-key-optimize_wrap div[data-testid="stButton"] button:hover {
  border-color: #fff; background: rgba(255,255,255,0.28);
}
div.st-key-pitch_card { gap: 0 !important; }
div.st-key-pitch_head { background: var(--panel); }

.bench-label {
  font-size: 0.62rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;
  color: #8b8a78; margin: 0 0 10px;
}

/* Chat: one bordered card holding the scrollable history and the input
   together, so it reads as a single chat box instead of two stacked
   pieces. Bubble shapes use Streamlit's own avatar test-ids to tell user
   turns from assistant turns. */
div.st-key-chat_card {
  border: 1px solid var(--border); border-radius: 12px; background: var(--panel); overflow: hidden;
}
div.st-key-chat_scroll { padding: 8px 10px 2px; min-height: 280px; }
div.st-key-chat_card div[data-testid="stChatInput"] {
  border-top: 1px solid var(--border); border-radius: 0;
}
div.st-key-chat_card div[data-testid="stChatInput"] textarea {
  background: var(--panel) !important; min-height: 44px !important;
}
div[data-testid="stChatMessage"] { padding: 3px 0; }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
  background: var(--turf-b); color: #fff; border-radius: 10px 10px 2px 10px;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
  background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px 10px 10px 2px;
}
div[data-testid="stChatMessageContent"] { padding: 8px 11px; font-size: 0.85rem; }
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
  padding: 62px 12px 30px;
  overflow: visible;
}
div.st-key-pitch::before {
  content: ""; position: absolute; left: 50%; top: 50%;
  width: 128px; height: 128px; border: 1.5px solid var(--line);
  border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none;
}
.pitch-halfway {
  position: absolute; left: 12px; right: 12px; top: 50%;
  border-top: 1.5px solid var(--line); transform: translateY(-50%); pointer-events: none;
}
div.st-key-bench { background: var(--bench); padding: 14px 12px 28px; overflow: visible; }
.slot-name {
  text-align: center; color: #fff; font-size: 0.72rem; font-weight: 700;
  line-height: 1.15; text-shadow: 0 1px 2px rgba(0,0,0,0.5); margin-top: 6px;
}
.slot-price {
  text-align: center; color: var(--gold); font-size: 0.64rem; font-weight: 700;
  font-variant-numeric: tabular-nums; text-shadow: 0 1px 2px rgba(0,0,0,0.5); margin-top: 1px;
}

/* Player slots render as circular, position-ringed "shirts" showing
   initials only, instead of plain rectangular buttons with full names.
   Streamlit wraps the marker div and the st.button in separate sibling
   "element-container" divs (not as direct siblings of each other), so
   :has() is needed to bridge from the marker up to its container, then
   across to the next container's button. */
div[data-testid="stElementContainer"]:has(.shirt-marker)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] {
  display: flex; justify-content: center;
}
div[data-testid="stElementContainer"]:has(.shirt-marker)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button {
  position: relative;
  border-radius: 50% !important;
  width: 46px !important; height: 46px !important; min-width: 46px !important;
  padding: 0 !important; margin: 0 auto !important;
  display: flex !important; align-items: center; justify-content: center;
  background: var(--bench) !important; color: #f2f1eb !important;
  font-size: 0.78rem !important; font-weight: 800 !important; line-height: 1 !important;
  letter-spacing: 0.02em; border: 2.5px solid #6b6b6b !important; white-space: normal !important;
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
  color: rgba(255,255,255,0.6) !important; box-shadow: none; font-size: 1rem !important;
}
/* Captain / vice badge: a small gold disc pinned to the shirt's corner,
   its letter set via CSS content since Streamlit buttons can't carry
   arbitrary child markup. */
div[data-testid="stElementContainer"]:has(.shirt-marker.captain)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::after,
div[data-testid="stElementContainer"]:has(.shirt-marker.vice)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::after {
  position: absolute; top: -6px; right: -6px; width: 17px; height: 17px;
  border-radius: 50%; background: var(--gold); color: var(--gold-ink);
  font-size: 0.6rem; font-weight: 800; display: flex; align-items: center; justify-content: center;
  border: 1.5px solid var(--turf-a);
}
div[data-testid="stElementContainer"]:has(.shirt-marker.captain)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::after { content: "C"; }
div[data-testid="stElementContainer"]:has(.shirt-marker.vice)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"] button::after { content: "V"; }
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
        st.markdown("<div style='height:2.15rem'></div>", unsafe_allow_html=True)
        if st.button("Build a new squad", use_container_width=True):
            st.session_state.squad = blank_squad()
            st.rerun()


def render_slot(squad, players, bucket, idx, position_hint):
    ids = squad[bucket]
    pid = ids[idx]
    shirt_label, name, price = "+", "Empty", None
    marker_classes = ["shirt-marker", "empty"]
    if pid is not None:
        row = players.set_index("id").loc[pid]
        name = row["web_name"]
        shirt_label = player_initials(name)
        price = row["now_cost"]
        pos = row["position"] if bucket == "bench_ids" else position_hint
        marker_classes = ["shirt-marker", pos.lower()]
        if pid == squad["captain_id"]:
            marker_classes.append("captain")
        elif pid == squad["vice_captain_id"]:
            marker_classes.append("vice")
    st.markdown(f"<div class='{' '.join(marker_classes)}'></div>", unsafe_allow_html=True)
    if st.button(shirt_label, key=f"{bucket}_{idx}", help=name if pid is not None else "Add player"):
        st.session_state.editing_slot = (bucket, idx, position_hint)
    st.markdown(f"<div class='slot-name'>{html.escape(str(name))}</div>", unsafe_allow_html=True)
    if price is not None:
        st.markdown(f"<div class='slot-price'>£{price}m</div>", unsafe_allow_html=True)


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


def render_toolbar(squad, players):
    # For an FPL-loaded squad, `bank` is already "leftover after the loaded squad"
    # and doesn't change as you reassign slots (that's the optimizer's job).
    # For a scratch squad there's no prior bank, so recompute from £100m - spend.
    remaining = squad["bank"] if squad["team_id"] else round(100.0 - squad_spend(squad, players), 1)
    locked = squad["team_id"] is not None

    toolbar_card = st.container(key="toolbar_card")
    stat_cols = toolbar_card.columns([0.8, 1, 1.6] + [1] * len(CHIP_NAMES))

    stat_cols[0].markdown(
        f"<div class='stat-label'>Cash</div><div class='stat-value'>£{remaining}m</div>",
        unsafe_allow_html=True,
    )

    with stat_cols[1]:
        st.markdown("<div class='stat-label'>Free transfers</div>", unsafe_allow_html=True)
        squad["free_transfers"] = st.selectbox(
            "Free transfers", list(range(6)), index=squad["free_transfers"],
            disabled=locked, label_visibility="collapsed", key="free_transfers_select",
        )

    chip_options = [None] + CHIP_NAMES
    with stat_cols[2]:
        st.markdown("<div class='stat-label'>Active chip</div>", unsafe_allow_html=True)
        squad["active_chip"] = st.selectbox(
            "Active chip", chip_options, index=chip_options.index(squad["active_chip"]),
            format_func=lambda c: "— none —" if c is None else CHIP_LABELS[c],
            label_visibility="collapsed", key="active_chip_select",
        )

    for col, name in zip(stat_cols[3:], CHIP_NAMES):
        with col:
            st.markdown(f"<div class='stat-label'>{CHIP_LABELS[name]}</div>", unsafe_allow_html=True)
            squad["chips_available"][name] = st.selectbox(
                CHIP_LABELS[name], [0, 1, 2], index=squad["chips_available"][name],
                disabled=locked, label_visibility="collapsed", key=f"chip_{name}",
            )


def run_optimizer(squad, players):
    with st.spinner("Solving..."):
        current = used_ids(squad) if squad["team_id"] else None
        result = optimize_squad(
            players, budget=total_budget(squad, players),
            free_transfers=squad["free_transfers"],
            active_chip=squad["active_chip"], current_ids=current,
        )
        apply_optimizer_result(squad, players, result)
    st.session_state.pop("team_summary_text", None)
    st.toast(
        f"{result['transfers_made']} transfer(s), -{result['points_penalty']} pt hit, "
        f"{result['predicted_total']} predicted pts (next 5 GWs)."
    )
    st.rerun()


def render_pitch():
    squad = st.session_state.squad
    players = load_players_df()
    st.markdown(PITCH_CSS, unsafe_allow_html=True)

    render_toolbar(squad, players)

    positions = slot_positions(squad["formation"])
    rows = [("GK", [0]), ("DEF", range(1, 1 + squad["formation"][0]))]
    rows.append(("MID", range(rows[-1][1].stop, rows[-1][1].stop + squad["formation"][1])))
    rows.append(("FWD", range(rows[-1][1].stop, rows[-1][1].stop + squad["formation"][2])))

    d, m, f = squad["formation"]
    pitch_card = st.container(key="pitch_card")
    with pitch_card:
        head = st.container(key="pitch_head")
        with head:
            head_l, head_r = st.columns([3, 1])
            head_l.markdown(
                f"<div class='pitch-title'>Starting XI <span class='formation-tag'>{d}-{m}-{f}</span></div>",
                unsafe_allow_html=True,
            )
            with head_r:
                with st.container(key="optimize_wrap"):
                    if st.button("Optimize", key="optimize_btn"):
                        run_optimizer(squad, players)

        with st.container(key="pitch"):
            st.markdown("<div class='pitch-halfway'></div>", unsafe_allow_html=True)
            for _, idx_range in rows:
                idx_range = list(idx_range)
                cols = st.columns(len(idx_range))
                for col, idx in zip(cols, idx_range):
                    with col:
                        render_slot(squad, players, "starting_ids", idx, positions[idx])

        with st.container(key="bench"):
            st.markdown("<div class='bench-label'>Bench</div>", unsafe_allow_html=True)
            bench_cols = st.columns(4)
            for col, idx in zip(bench_cols, range(4)):
                with col:
                    render_slot(squad, players, "bench_ids", idx, "SUB")

    render_slot_editor(squad, players)


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

    chat_card = st.container(key="chat_card")
    history = chat_card.container(height=280, border=False, key="chat_scroll")
    for message in st.session_state.messages:
        with history.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := chat_card.chat_input("Ask about your team"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with history.chat_message("user"):
            st.markdown(prompt)

        with history.chat_message("assistant"):
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
    tab_team, tab_players = st.tabs(["Team", "Players"])
    with tab_team:
        pitch_col, side_col = st.columns([3, 1])
        with pitch_col:
            render_pitch()
        with side_col:
            render_side_panel()
    with tab_players:
        render_players_tab()
