"""
Interactive dashboard for Fantasy Premier League player analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import os
from rag_system import PlayerNewsRAG
from data_fetcher import FantasyDataFetcher

# Set up project root path for file resolution
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


# Page configuration
st.set_page_config(
    page_title="Fantasy PL Manager",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premier League Color Palette
PL_BLUE = "#003399"
PL_WHITE = "#FFFFFF"
PL_GOLD = "#FFB500"
PL_DARK = "#1a1a1a"
PL_LIGHT_GRAY = "#f5f5f5"

# Custom CSS styling
st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {PL_BLUE} 0%, {PL_DARK} 100%);
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {PL_WHITE};
    }}

    /* Header styling */
    h1 {{
        color: {PL_BLUE};
        font-weight: 800;
        letter-spacing: -0.5px;
    }}

    h2 {{
        color: {PL_BLUE};
        font-weight: 700;
        border-bottom: 3px solid {PL_GOLD};
        padding-bottom: 10px;
    }}

    h3 {{
        color: {PL_DARK};
        font-weight: 600;
    }}

    /* Metric styling */
    [data-testid="metric-container"] {{
        background: linear-gradient(to right, {PL_WHITE}, {PL_LIGHT_GRAY});
        border-left: 4px solid {PL_BLUE};
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 51, 153, 0.1);
    }}

    /* Tabs styling */
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: 2px solid {PL_GOLD};
    }}

    [data-testid="stTabs"] [role="tab"] {{
        color: {PL_DARK};
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }}

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: linear-gradient(to bottom, {PL_BLUE}, {PL_DARK});
        color: {PL_WHITE};
        border-bottom: 3px solid {PL_GOLD};
    }}

    /* Button styling */
    .stButton > button {{
        background: linear-gradient(to right, {PL_BLUE}, {PL_DARK});
        color: {PL_WHITE};
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 51, 153, 0.3);
    }}

    /* Expander styling */
    [data-testid="stExpander"] {{
        border: 1px solid {PL_GOLD};
        border-radius: 8px;
    }}

    [data-testid="stExpander"] summary {{
        color: {PL_BLUE};
        font-weight: 600;
    }}

    /* Dataframe styling */
    [data-testid="stDataFrame"] {{
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Select and slider styling */
    .stSelectbox div[data-baseweb="select"] {{
        border-color: {PL_BLUE};
    }}

    .stSlider {{
        padding: 20px 0;
    }}

    /* Info/Warning boxes */
    [data-testid="stAlert"] {{
        border-radius: 8px;
        border-left: 4px solid {PL_GOLD};
    }}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_player_data():
    """Load player data with sentiment scores."""
    try:
        file_path = DATA_DIR / 'players_with_sentiment.csv'
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run data_fetcher.py and sentiment_analyzer.py first")
        return None


@st.cache_data
def load_sentiment_data():
    """Load detailed sentiment analysis."""
    try:
        file_path = DATA_DIR / 'sentiment_analysis.json'
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return []


@st.cache_data
def load_predictions_data():
    """Load prediction data."""
    try:
        file_path = DATA_DIR / 'predictions.json'
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}


@st.cache_data(ttl=3600)
def load_fixture_lookup():
    """Fetch a lightweight player->next fixture lookup for UI fallbacks."""
    try:
        fetcher = FantasyDataFetcher()
        players = fetcher.fetch_players()
        lookup = {}
        for player in players:
            lookup[player['name']] = {
                'opp_team_name': player.get('opp_team_name') or 'TBD',
                'was_home': player.get('was_home'),
                'points_per_game': float(player.get('points_per_game', 0) or 0),
            }
        return lookup
    except Exception:
        return {}


def get_data_version() -> str:
    """Build a cache key from data and index files so RAG reloads after updates."""
    files = [
        DATA_DIR / 'players.json',
        DATA_DIR / 'news.json',
        DATA_DIR / 'predictions.json',
        PROJECT_ROOT / 'vector_db' / 'index.faiss',
        PROJECT_ROOT / 'vector_db' / 'metadata.pkl',
    ]

    version_parts = []
    for path in files:
        if path.exists():
            stat = path.stat()
            version_parts.append(f"{path.name}:{int(stat.st_mtime)}:{int(stat.st_size)}")
        else:
            version_parts.append(f"{path.name}:missing")
    return "|".join(version_parts)


@st.cache_resource
def load_rag_system(data_version: str):
    """Load RAG system with FAISS, building index if needed."""
    try:
        _ = data_version
        rag = PlayerNewsRAG()
        if not rag.load_data():
            return None  # No data available
        
        # Try to load existing index
        if not rag._load_index():
            # Index doesn't exist - try to build it
            try:
                with st.spinner("Building vector database (first time only)..."):
                    rag.create_vector_db()
                return rag
            except Exception as e:
                print(f"Failed to build vector database: {e}")
                return None  # Building failed

        index_stats = rag.get_index_stats()
        needs_rebuild = (
            rag.is_index_stale()
            or index_stats.get('total_docs', 0) == 0
            or (len(rag.news_data) > 0 and index_stats.get('news_docs', 0) == 0)
            or (len(rag.predictions_data) > 0 and index_stats.get('prediction_docs', 0) == 0)
        )

        if needs_rebuild:
            with st.spinner("Refreshing vector database from latest news + predictions..."):
                rag.create_vector_db()
        
        return rag  # Index loaded successfully
    except Exception as e:
        print(f"RAG system error: {e}")
        return None


def main():
    """Main dashboard function with chatbot-first layout and sidebar navigation."""

    st.markdown(f"""
    <div style="background: linear-gradient(120deg, #0a2d6b 0%, #123f8e 45%, #1b5cb8 100%);
                padding: 26px 28px; border-radius: 14px; margin-bottom: 24px; border: 1px solid #2158aa;">
        <h1 style="color: {PL_WHITE}; margin: 0 0 6px 0; font-size: 34px;">Fantasy PL Intelligence Workspace</h1>
        <p style="color: #d7e6ff; margin: 0; font-size: 15px;">
            Chat-first assistant with linked views for predictions, sentiment intelligence, and player scouting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_player_data()
    if df is None or df.empty:
        st.stop()

    predictions_data = load_predictions_data()
    fixture_lookup = load_fixture_lookup()

    if predictions_data:
        pred_df = pd.DataFrame(predictions_data.values())
        prediction_cols = [
            'player_name', 'predicted_total_points', 'predicted_points_per_match',
            'form_trend', 'next_game_predicted_points', 'next_game_opponent',
            'next_game_difficulty'
        ]
        available_cols = [c for c in prediction_cols if c in pred_df.columns]
        if 'player_name' in available_cols:
            df = df.merge(pred_df[available_cols], left_on='name', right_on='player_name', how='left')
            df = df.drop('player_name', axis=1)

    for col, default in [
        ('predicted_total_points', 0.0),
        ('predicted_points_per_match', 0.0),
        ('form_trend', 'unknown'),
        ('next_game_predicted_points', 0.0),
        ('next_game_opponent', 'Unknown'),
        ('next_game_difficulty', 'Unknown'),
        ('normalized_score', np.nan),
        ('article_count', 0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    # Fixture fallback for stale prediction files
    df['fixture_opp_fallback'] = df['name'].map(lambda n: fixture_lookup.get(n, {}).get('opp_team_name', 'TBD'))
    df['fixture_ppg_fallback'] = df['name'].map(lambda n: fixture_lookup.get(n, {}).get('points_per_game', 0.0))

    unknown_mask = df['next_game_opponent'].astype(str).str.lower().isin(['unknown', 'none', 'nan', ''])
    df.loc[unknown_mask, 'next_game_opponent'] = df.loc[unknown_mask, 'fixture_opp_fallback']

    zero_next_mask = (pd.to_numeric(df['next_game_predicted_points'], errors='coerce').fillna(0) <= 0)
    ppm_vals = pd.to_numeric(df['predicted_points_per_match'], errors='coerce').fillna(0)
    ppg_vals = pd.to_numeric(df['fixture_ppg_fallback'], errors='coerce').fillna(0)
    blended = (0.7 * ppm_vals + 0.3 * ppg_vals).clip(lower=0)
    df.loc[zero_next_mask, 'next_game_predicted_points'] = blended[zero_next_mask]
    df = df.drop(columns=['fixture_opp_fallback', 'fixture_ppg_fallback'], errors='ignore')

    # Sidebar navigation + global filters
    with st.sidebar:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.08); padding: 14px; border-radius: 10px; margin-bottom: 14px;">
            <h3 style="margin: 0; color: #fff; font-size: 18px;">Workspace Navigation</h3>
            <p style="margin: 4px 0 0 0; color: #dbe8ff; font-size: 12px;">Switch views without losing context.</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Go to",
            [
                "Chat Assistant",
                "Player Predictions",
                "Sentiment Analysis",
                "Analytics",
                "Player Table",
                "About",
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Filters")

        all_teams = sorted(df['team'].unique())
        selected_teams = st.multiselect("Teams", options=all_teams, default=all_teams)

        all_positions = sorted(df['position'].unique())
        selected_positions = st.multiselect("Positions", options=all_positions, default=all_positions)

        price_range = st.slider(
            "Price Range (£M)",
            float(df['now_cost'].min()),
            float(df['now_cost'].max()),
            (float(df['now_cost'].min()), float(df['now_cost'].max())),
            0.1,
        )

        show_only_scored = st.checkbox("Only players with sentiment score", value=False)
        min_articles = st.slider("Minimum articles", 0, 100, 5, disabled=not show_only_scored)

    filtered_df = df[
        (df['team'].isin(selected_teams))
        & (df['position'].isin(selected_positions))
        & (df['now_cost'] >= price_range[0])
        & (df['now_cost'] <= price_range[1])
    ]

    if show_only_scored:
        filtered_df = filtered_df[
            filtered_df['normalized_score'].notna() & (filtered_df['article_count'] >= min_articles)
        ]

    if page == "Chat Assistant":
        st.markdown("## Chat Assistant")
        st.caption("Ask natural-language questions and get evidence from live player news and prediction context.")

        players_exists = (DATA_DIR / 'players.json').exists()
        news_exists = (DATA_DIR / 'news.json').exists()
        predictions_exists = (DATA_DIR / 'predictions.json').exists()
        has_data = predictions_exists or news_exists

        with st.expander("System Status", expanded=False):
            st.write(f"Project Root: {PROJECT_ROOT}")
            st.write(f"Data Directory: {DATA_DIR}")
            st.write("Players JSON available" if players_exists else "Players JSON missing (auto-fetch enabled)")
            st.write("News JSON available" if news_exists else "News JSON missing")
            st.write("Predictions JSON available" if predictions_exists else "Predictions JSON missing")

        if not has_data:
            st.warning("No news or prediction data source is available yet. Run the pipeline to populate data.")
        else:
            data_version = get_data_version()
            rag = load_rag_system(data_version)

            if rag is None:
                st.error("RAG index is not ready. Refresh or rerun the data pipeline.")
            else:
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = [
                        {
                            'role': 'assistant',
                            'content': (
                                "Ask about captain picks, injuries, fixture-driven punts, or value picks. "
                                "I will answer using your latest news and prediction documents."
                            )
                        }
                    ]

                for msg in st.session_state.chat_history:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])

                prompt = st.chat_input("Ask about players, form, fixtures, and recommendations")
                if prompt:
                    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

                    with st.spinner("Searching and drafting response..."):
                        results = rag.query(prompt, n_results=6)
                        if not results:
                            answer = "I could not find relevant documents for that query. Try naming a player, position, or gameweek context."
                        else:
                            news_hits = [r for r in results if r.get('type') == 'news'][:3]
                            pred_hits = [r for r in results if r.get('type') == 'prediction'][:3]

                            summary_lines = []
                            if pred_hits:
                                top_pred = pred_hits[0]
                                summary_lines.append(
                                    f"Top prediction signal: {top_pred['player_name']} ({top_pred['team']}) with "
                                    f"next GW projection {top_pred.get('next_game_predicted_points', 0):.2f} and "
                                    f"fixture vs {top_pred.get('next_game_opponent', 'Unknown')}."
                                )
                            if news_hits:
                                strong_news = news_hits[0]
                                summary_lines.append(
                                    f"Latest news driver: {strong_news['player_name']} - {strong_news.get('title', 'No title')} "
                                    f"(sentiment {strong_news.get('sentiment_score', 50):.0f}/100)."
                                )

                            picks = []
                            for r in pred_hits[:3]:
                                picks.append(
                                    f"{r['player_name']} ({r['team']}) | Next GW {r.get('next_game_predicted_points', 0):.2f} | "
                                    f"{r.get('next_game_opponent', 'Unknown')}"
                                )

                            answer = "\n\n".join([
                                " ".join(summary_lines) if summary_lines else "I found relevant context.",
                                "Shortlist:\n" + "\n".join([f"- {p}" for p in picks]) if picks else ""
                            ]).strip()

                    st.session_state.chat_history.append({'role': 'assistant', 'content': answer})

                    with st.expander("Sources used", expanded=False):
                        for i, r in enumerate(results, 1):
                            if r.get('type') == 'news':
                                st.markdown(
                                    f"**{i}. News** - {r.get('player_name')} | {r.get('title', 'Untitled')} | "
                                    f"Sentiment {r.get('sentiment_score', 50):.0f}/100"
                                )
                            else:
                                st.markdown(
                                    f"**{i}. Prediction** - {r.get('player_name')} | Next GW "
                                    f"{r.get('next_game_predicted_points', 0):.2f} | "
                                    f"{r.get('next_game_opponent', 'Unknown')}"
                                )

                    st.rerun()

    elif page == "Player Predictions":
        st.markdown("## Player Predictions")
        pred_display_df = filtered_df[filtered_df['predicted_total_points'] > 0].copy()
        if pred_display_df.empty:
            st.info("No prediction data in current filter.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Players", len(pred_display_df))
            with col2:
                st.metric("Avg Predicted Total", f"{pred_display_df['predicted_total_points'].mean():.1f}")
            with col3:
                st.metric("Best Next GW", f"{pred_display_df['next_game_predicted_points'].max():.2f}")
            with col4:
                st.metric("Improving Form", int((pred_display_df['form_trend'] == 'improving').sum()))

            sort_by_pred = st.selectbox(
                "Sort by",
                [
                    "next_game_predicted_points",
                    "predicted_total_points",
                    "predicted_points_per_match",
                    "now_cost",
                    "total_points",
                ],
                format_func=lambda x: {
                    "next_game_predicted_points": "Next Gameweek Points",
                    "predicted_total_points": "Predicted Total Points",
                    "predicted_points_per_match": "Predicted PPM",
                    "now_cost": "Price",
                    "total_points": "Current Points",
                }[x],
            )
            ascending_pred = st.checkbox("Ascending", value=False, key="pred_sort_asc")

            pred_table = pred_display_df.sort_values(sort_by_pred, ascending=ascending_pred)[[
                'name', 'team', 'position', 'now_cost', 'predicted_total_points',
                'predicted_points_per_match', 'next_game_predicted_points',
                'next_game_opponent', 'next_game_difficulty', 'form_trend', 'total_points'
            ]].copy()

            pred_table.columns = [
                'Player', 'Team', 'Position', 'Price', 'Pred Total', 'Pred PPM',
                'Next GW Pred', 'Next Opponent', 'Fixture Difficulty', 'Form', 'Current Pts'
            ]
            pred_table['Pred Total'] = pred_table['Pred Total'].map(lambda x: f"{x:.1f}")
            pred_table['Pred PPM'] = pred_table['Pred PPM'].map(lambda x: f"{x:.2f}")
            pred_table['Next GW Pred'] = pred_table['Next GW Pred'].map(lambda x: f"{x:.2f}")
            st.dataframe(pred_table, use_container_width=True, hide_index=True, height=620)

    elif page == "Sentiment Analysis":
        st.markdown("## Sentiment Analysis")
        scored_df = filtered_df[filtered_df['normalized_score'].notna()].copy()
        if scored_df.empty:
            st.info("No sentiment scores in current filter.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.box(
                    scored_df,
                    x='team',
                    y='normalized_score',
                    color='team',
                    title="Sentiment Distribution by Team"
                )
                fig.update_layout(showlegend=False, height=420, template='plotly_white')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    scored_df,
                    x='normalized_score',
                    y='total_points',
                    color='position',
                    size='now_cost',
                    hover_data=['name', 'team'],
                    title="Sentiment vs Current Points"
                )
                fig.update_layout(height=420, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

            best = scored_df.nlargest(10, 'normalized_score')[['name', 'team', 'normalized_score']]
            worst = scored_df.nsmallest(10, 'normalized_score')[['name', 'team', 'normalized_score']]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Most Positive")
                for _, row in best.iterrows():
                    st.write(f"{row['name']} ({row['team']}) - {row['normalized_score']:.1f}/100")
            with c2:
                st.markdown("### Most Negative")
                for _, row in worst.iterrows():
                    st.write(f"{row['name']} ({row['team']}) - {row['normalized_score']:.1f}/100")

    elif page == "Analytics":
        st.markdown("## Analytics")
        pred_analytics_df = filtered_df[filtered_df['predicted_total_points'] > 0].copy()
        if pred_analytics_df.empty:
            st.info("No prediction analytics available in current filter.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                top_20 = pred_analytics_df.nlargest(20, 'predicted_total_points')
                fig = px.bar(
                    top_20,
                    x='predicted_total_points',
                    y='name',
                    color='position',
                    orientation='h',
                    title="Top 20 Predicted Total Points"
                )
                fig.update_layout(height=520, template='plotly_white', yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(
                    pred_analytics_df,
                    x='position',
                    y='predicted_total_points',
                    color='position',
                    title="Predicted Points Distribution by Position"
                )
                fig.update_layout(height=520, showlegend=False, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

            fig = px.scatter(
                pred_analytics_df,
                x='now_cost',
                y='predicted_total_points',
                color='position',
                size='total_points',
                hover_data=['name', 'team', 'form_trend'],
                title="Price vs Predicted Performance"
            )
            fig.update_layout(height=430, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Player Table":
        st.markdown("## Player Table")
        sort_by = st.selectbox(
            "Sort by",
            ["normalized_score", "total_points", "now_cost", "form", "selected_by_percent", "article_count"],
            format_func=lambda x: {
                "normalized_score": "Sentiment Score",
                "total_points": "Total Points",
                "now_cost": "Price",
                "form": "Form",
                "selected_by_percent": "Selected By %",
                "article_count": "Article Count",
            }[x],
        )
        ascending = st.checkbox("Ascending", value=False, key="table_asc")
        display_df = filtered_df.sort_values(sort_by, ascending=ascending)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Players", len(display_df))
        with m2:
            st.metric("With Sentiment", int(display_df['normalized_score'].notna().sum()))
        with m3:
            avg_sentiment = display_df['normalized_score'].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.1f}/100" if pd.notna(avg_sentiment) else "N/A")
        with m4:
            st.metric("Total Articles", f"{int(display_df['article_count'].sum()):,}")

        table = display_df[[
            'name', 'team', 'position', 'now_cost', 'total_points',
            'form', 'normalized_score', 'article_count'
        ]].copy()
        table.columns = ['Player', 'Team', 'Position', 'Price (£M)', 'Points', 'Form', 'Sentiment', 'Articles']
        table['Sentiment'] = table['Sentiment'].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        table['Articles'] = table['Articles'].fillna(0).astype(int)
        st.dataframe(table, use_container_width=True, hide_index=True, height=680)

    else:
        st.markdown("## About")
        st.markdown(
            """
            This workspace combines player statistics, sentiment intelligence, and machine-learning predictions.

            - Chat Assistant: natural-language scouting grounded in RAG retrieval.
            - Player Predictions: sortable per-player forecast sheet including next gameweek signals.
            - Sentiment Analysis: media-driven player momentum and risk context.
            - Analytics: visual exploration of value, upside, and positional trends.
            """
        )



if __name__ == "__main__":
    main()
