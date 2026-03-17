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
    """Main dashboard function."""

    # Enhanced Header with Premier League branding
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {PL_BLUE} 0%, {PL_DARK} 100%);
                padding: 30px; border-radius: 12px; margin-bottom: 30px;">
        <h1 style="color: {PL_WHITE}; margin: 0 0 10px 0;">
            ⚽ Fantasy Premier League Manager
        </h1>
        <p style="color: {PL_GOLD}; font-size: 16px; margin: 0; font-weight: 500;">
            AI-powered player analysis with sentiment scoring, ML predictions & news insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_player_data()
    
    if df is None or df.empty:
        st.stop()
    
    sentiment_data = load_sentiment_data()
    predictions_data = load_predictions_data()
    
    # Merge predictions into dataframe
    if predictions_data:
        pred_df = pd.DataFrame(predictions_data.values())
        prediction_cols = [
            'player_name', 'predicted_total_points', 'predicted_points_per_match',
            'form_trend', 'next_game_predicted_points', 'next_game_opponent',
            'next_game_difficulty'
        ]
        available_cols = [c for c in prediction_cols if c in pred_df.columns]
        if 'player_name' in available_cols:
            df = df.merge(
                pred_df[available_cols],
                left_on='name',
                right_on='player_name',
                how='left'
            )
            df = df.drop('player_name', axis=1)
    else:
        df['predicted_total_points'] = 0
        df['predicted_points_per_match'] = 0
        df['form_trend'] = 'unknown'
        df['next_game_predicted_points'] = 0
        df['next_game_opponent'] = 'Unknown'
        df['next_game_difficulty'] = 'Unknown'

    for col, default in [
        ('next_game_predicted_points', 0.0),
        ('next_game_opponent', 'Unknown'),
        ('next_game_difficulty', 'Unknown'),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    
    # Main panel filters (sidebar removed)
    with st.expander("🎛️ Player Filters", expanded=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            all_teams = sorted(df['team'].unique())
            selected_teams = st.multiselect(
                "Teams",
                options=all_teams,
                default=all_teams,
            )

        with c2:
            all_positions = sorted(df['position'].unique())
            selected_positions = st.multiselect(
                "Positions",
                options=all_positions,
                default=all_positions,
            )

        with c3:
            price_range = st.slider(
                "Price Range (£M)",
                float(df['now_cost'].min()),
                float(df['now_cost'].max()),
                (float(df['now_cost'].min()), float(df['now_cost'].max())),
                0.1,
            )

        s1, s2 = st.columns(2)
        with s1:
            show_only_scored = st.checkbox("Only players with sentiment score", value=False)
        with s2:
            min_articles = st.slider("Minimum articles", 0, 100, 50, disabled=not show_only_scored)
    
    # Apply filters
    filtered_df = df[
        (df['team'].isin(selected_teams)) &
        (df['position'].isin(selected_positions)) &
        (df['now_cost'] >= price_range[0]) &
        (df['now_cost'] <= price_range[1])
    ]
    
    if show_only_scored:
        filtered_df = filtered_df[
            (filtered_df['has_enough_articles'] == True) &
            (filtered_df['article_count'] >= min_articles)
        ]
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Player Table",
        "🔮 Predictions",
        "📈 Analytics",
        "🔍 RAG Search",
        "ℹ️ About"
    ])

    with tab1:
        st.markdown(f"<h2>📊 Player Data</h2>", unsafe_allow_html=True)

        # Sort options
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["normalized_score", "total_points", "now_cost", "form", "selected_by_percent", "article_count"],
                format_func=lambda x: {
                    "normalized_score": "Sentiment Score",
                    "total_points": "Total Points",
                    "now_cost": "Price",
                    "form": "Form",
                    "selected_by_percent": "Selected By %",
                    "article_count": "Article Count"
                }[x],
                label_visibility="collapsed"
            )
        with col2:
            ascending = st.checkbox("Ascending", value=False)

        # Sort dataframe
        display_df = filtered_df.sort_values(sort_by, ascending=ascending)

        # Display metrics with Premier League styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Players", len(display_df))
        with col2:
            scored_players = len(display_df[display_df['has_enough_articles'] == True])
            st.metric("💬 With Sentiment", scored_players)
        with col3:
            avg_sentiment = display_df['normalized_score'].mean()
            if pd.notna(avg_sentiment):
                st.metric("⭐ Avg Sentiment", f"{avg_sentiment:.1f}/100")
            else:
                st.metric("⭐ Avg Sentiment", "N/A")
        with col4:
            total_articles = display_df['article_count'].sum()
            st.metric("📰 Total Articles", f"{int(total_articles):,}")
        
        # Display table
        display_columns = [
            'name', 'team', 'position', 'now_cost', 'total_points', 
            'form', 'normalized_score', 'article_count'
        ]
        
        # Rename columns for display
        column_names = {
            'name': 'Player',
            'team': 'Team',
            'position': 'Position',
            'now_cost': 'Price (£M)',
            'total_points': 'Points',
            'form': 'Form',
            'normalized_score': 'Sentiment',
            'article_count': 'Articles'
        }
        
        display_table = display_df[display_columns].copy()
        display_table = display_table.rename(columns=column_names)
        
        # Format numbers
        display_table['Sentiment'] = display_table['Sentiment'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        display_table['Articles'] = display_table['Articles'].fillna(0).astype(int)
        
        st.dataframe(
            display_table,
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    with tab2:
        st.markdown(f"<h2>🔮 Player Performance Predictions</h2>", unsafe_allow_html=True)

        if not predictions_data:
            st.warning("⚠️ No predictions available. Please run: `python src/predictor.py`")
        else:
            # Sort options for predictions
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by_pred = st.selectbox(
                    "Sort predictions by",
                    ["next_game_predicted_points", "predicted_total_points", "predicted_points_per_match", "now_cost", "total_points"],
                    format_func=lambda x: {
                        "next_game_predicted_points": "Next Gameweek Points",
                        "predicted_total_points": "Predicted Total Points",
                        "predicted_points_per_match": "Predicted PPM",
                        "now_cost": "Price",
                        "total_points": "Current Points"
                    }[x],
                    label_visibility="collapsed"
                )
            with col2:
                ascending_pred = st.checkbox("Ascending ", value=False, key="pred_asc")

            # Sort by predictions
            pred_display_df = filtered_df[filtered_df['predicted_total_points'] > 0].sort_values(
                sort_by_pred, ascending=ascending_pred
            )

            # Display prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✨ With Predictions", len(pred_display_df))
            with col2:
                avg_pred = pred_display_df['predicted_total_points'].mean()
                st.metric("📊 Avg Predicted Points", f"{avg_pred:.1f}")
            with col3:
                top_next = pred_display_df['next_game_predicted_points'].max()
                st.metric("🎯 Best Next GW", f"{top_next:.2f}")
            with col4:
                improving = len(pred_display_df[pred_display_df['form_trend'] == 'improving'])
                st.metric("📈 Players Improving", improving)
            
            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏅 Top 20 Predicted Points")
                top_20 = pred_display_df.nlargest(20, 'predicted_total_points')
                fig = px.bar(
                    top_20,
                    x='predicted_total_points',
                    y='name',
                    color='position',
                    title="Highest Predicted Total Points",
                    orientation='h',
                    labels={'predicted_total_points': 'Predicted Points', 'name': 'Player'}
                )
                fig.update_layout(
                    height=500,
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white',
                    colorway=[PL_BLUE, PL_GOLD, '#E8423C', '#7CB342']
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📊 Predicted Points by Position")
                fig = px.box(
                    pred_display_df,
                    x='position',
                    y='predicted_total_points',
                    color='position',
                    title="Point Distribution by Position"
                )
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    template='plotly_white',
                    colorway=[PL_BLUE, PL_GOLD, '#E8423C', '#7CB342']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Value analysis
            st.subheader("💰 Value Analysis: Price vs Predicted Points")
            fig = px.scatter(
                pred_display_df,
                x='now_cost',
                y='predicted_total_points',
                color='position',
                size='total_points',
                hover_data=['name', 'team', 'form_trend'],
                title="Price vs Predicted Performance (size = current points)"
            )
            fig.update_layout(
                height=400,
                template='plotly_white',
                colorway=[PL_BLUE, PL_GOLD, '#E8423C', '#7CB342']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions table
            st.subheader("Player Predictions")
            pred_columns = [
                'name', 'team', 'position', 'now_cost', 'predicted_total_points',
                'predicted_points_per_match', 'next_game_predicted_points',
                'next_game_opponent', 'next_game_difficulty', 'form_trend', 'total_points'
            ]
            
            pred_table = pred_display_df[pred_columns].copy()
            pred_table.columns = [
                'Player', 'Team', 'Pos', 'Price', 'Pred Total', 'Pred PPM', 'Next GW Pred',
                'Next Opponent', 'Fixture Diff',
                'Form', 'Current Pts'
            ]
            
            # Format numbers
            pred_table['Pred Total'] = pred_table['Pred Total'].apply(lambda x: f"{x:.1f}")
            pred_table['Pred PPM'] = pred_table['Pred PPM'].apply(lambda x: f"{x:.2f}")
            pred_table['Next GW Pred'] = pred_table['Next GW Pred'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                pred_table,
                use_container_width=True,
                height=400,
                hide_index=True
            )
    
    with tab3:
        st.markdown(f"<h2>📈 Analytics Dashboard</h2>", unsafe_allow_html=True)

        # Only show analytics for players with sentiment scores
        scored_df = filtered_df[filtered_df['has_enough_articles'] == True].copy()

        if len(scored_df) == 0:
            st.warning("⚠️ No players with sentiment scores in current filter. Adjust filters to see analytics.")
        else:
            # Sentiment distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Sentiment Distribution by Team")
                fig = px.box(
                    scored_df,
                    x='team',
                    y='normalized_score',
                    color='team',
                    title="Sentiment Scores by Team"
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("⭐ Sentiment vs Points")
                fig = px.scatter(
                    scored_df,
                    x='normalized_score',
                    y='total_points',
                    color='position',
                    size='now_cost',
                    hover_data=['name', 'team'],
                    title="Sentiment Score vs Total Points"
                )
                fig.update_layout(
                    height=400,
                    template='plotly_white',
                    colorway=[PL_BLUE, PL_GOLD, '#E8423C', '#7CB342']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Top performers
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"<h3 style='color: {PL_BLUE};'>🔥 Most Positive Sentiment</h3>", unsafe_allow_html=True)
                top_positive = scored_df.nlargest(10, 'normalized_score')[
                    ['name', 'team', 'normalized_score', 'total_points']
                ]
                for idx, row in top_positive.iterrows():
                    st.write(f"**{row['name']}** ({row['team']}) · {row['normalized_score']:.1f}/100")

            with col2:
                st.markdown(f"<h3 style='color: {PL_BLUE};'>📉 Most Negative Sentiment</h3>", unsafe_allow_html=True)
                top_negative = scored_df.nsmallest(10, 'normalized_score')[
                    ['name', 'team', 'normalized_score', 'total_points']
                ]
                for idx, row in top_negative.iterrows():
                    st.write(f"**{row['name']}** ({row['team']}) · {row['normalized_score']:.1f}/100")

            # Sentiment by position
            st.subheader("📍 Average Sentiment by Position")
            position_sentiment = scored_df.groupby('position')['normalized_score'].mean().reset_index()
            fig = px.bar(
                position_sentiment,
                x='position',
                y='normalized_score',
                title="Average Sentiment Score by Position",
                color='normalized_score',
                color_continuous_scale=[[0, '#E8423C'], [0.5, '#FFB500'], [1, '#7CB342']]
            )
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

            # Price vs Sentiment
            st.subheader("💰 Value Analysis: Price vs Sentiment")
            fig = px.scatter(
                scored_df,
                x='now_cost',
                y='normalized_score',
                color='position',
                size='total_points',
                hover_data=['name', 'team'],
                title="Price vs Sentiment (size = total points)"
            )
            fig.update_layout(
                height=400,
                template='plotly_white',
                colorway=[PL_BLUE, PL_GOLD, '#E8423C', '#7CB342']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown(f"<h2>🔍 RAG Search - Ask Questions About Players</h2>", unsafe_allow_html=True)
        st.markdown("Search through player news and predictions using natural language. Ask anything about player form, injuries, or predictions!")

        # Check if required data files exist (using absolute paths)
        players_exists = (DATA_DIR / 'players.json').exists()
        news_exists = (DATA_DIR / 'news.json').exists()
        predictions_exists = (DATA_DIR / 'predictions.json').exists()
        has_data = players_exists and news_exists and predictions_exists

        # Debug info for developers
        with st.expander("🔧 Debug Info (System Status)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Project Root:** {PROJECT_ROOT}")
                st.write(f"**Data Directory:** {DATA_DIR}")
            with col2:
                st.write(f"✅ Players JSON" if players_exists else "❌ Players JSON missing")
                st.write(f"✅ News JSON ({(DATA_DIR / 'news.json').stat().st_size / (1024*1024):.1f}MB)" if news_exists else "❌ News JSON missing")
                st.write(f"✅ Predictions JSON" if predictions_exists else "❌ Predictions JSON missing")

        if not has_data:
            st.info("📊 **RAG Search Not Ready Yet**")
            st.markdown(f"""
            <div style="background: {PL_GOLD}20; padding: 15px; border-radius: 8px; border-left: 4px solid {PL_GOLD};">
            The pipeline data hasn't been generated yet. Required files:

            - 🔍 Player data (from FPL API)
            - 🔍 News articles (from Google News)
            - 🔍 Predictions (from ML models)

            **To run the pipeline manually:**
            ```bash
            python run_pipeline.py
            ```

            The daily automated pipeline will generate this data automatically!
            </div>
            """, unsafe_allow_html=True)
        else:
            # Load RAG system
            data_version = get_data_version()
            rag = load_rag_system(data_version)

            if rag is None:
                st.warning("⚠️ **Vector Database Build Failed**")
                st.markdown(f"""
                The data files exist but the vector database couldn't be created.

                This might be due to:
                - Insufficient memory (requires ~500MB RAM)
                - Missing dependencies (FAISS)
                - Corrupted data files

                Try refreshing the page or wait for the next automated update.
                """)
            else:
                # Query section
                st.markdown(f"<h3 style='color: {PL_BLUE};'>What would you like to know?</h3>", unsafe_allow_html=True)

                # Query input
                query = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., 'Which defenders are in great form?' or 'Who are the highest value midfielders?'",
                    label_visibility="collapsed"
                )

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    player_filter = st.selectbox(
                        "Filter by player",
                        ["All players"] + sorted(df['name'].unique().tolist()),
                        label_visibility="collapsed"
                    )
                with col2:
                    doc_type = st.selectbox(
                        "Document type",
                        ["All", "News", "Predictions"],
                        label_visibility="collapsed"
                    )
                with col3:
                    n_results = st.slider("Results", 1, 20, 5)

                if query:
                    with st.spinner("🔍 Searching database..."):
                        # Execute query
                        filter_player = None if player_filter == "All players" else player_filter
                        filter_type = None if doc_type == "All" else doc_type.lower()[:-1]  # Remove 's'
                        results = rag.query(
                            query,
                            n_results=n_results,
                            player_filter=filter_player,
                            doc_type_filter=filter_type
                        )

                        st.success(f"✨ Found {len(results)} relevant results")

                        # Display results with styling
                        for i, result in enumerate(results, 1):
                            result_type = result.get('type', 'news')

                            if result_type == 'news':
                                with st.expander(
                                    f"#{i} 📰 {result['player_name']} · {result['title'][:55]}...",
                                    expanded=i==1
                                ):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{result['player_name']}** ({result['team']})")
                                        st.markdown(f"__{result['title']}__")
                                    with col2:
                                        sentiment = result.get('sentiment_score', 0)
                                        if sentiment > 60:
                                            st.markdown(f"<span style='color: #7CB342;'>⭐ {sentiment:.0f}/100</span>", unsafe_allow_html=True)
                                        elif sentiment > 40:
                                            st.markdown(f"<span style='color: {PL_GOLD};'>⭐ {sentiment:.0f}/100</span>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<span style='color: #E8423C;'>⭐ {sentiment:.0f}/100</span>", unsafe_allow_html=True)

                                    st.caption(f"📌 {result.get('source', 'Unknown')} · {result.get('published', 'Unknown')}")
                                    st.markdown(result['content'][:800] + "...")
                                    if result.get('link'):
                                        st.markdown(f"[🔗 Read full article →]({result['link']})")
                            else:
                                with st.expander(
                                    f"#{i} 🔮 {result['player_name']} · Performance Prediction",
                                    expanded=i==1
                                ):
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{result['player_name']}** ({result['team']})")
                                        st.markdown(f"__Performance Forecast__")
                                    with col2:
                                        trend = result.get('form_trend', 'stable')
                                        if trend == 'improving':
                                            st.markdown(f"<span style='color: #7CB342;'>📈 Improving</span>", unsafe_allow_html=True)
                                        elif trend == 'declining':
                                            st.markdown(f"<span style='color: #E8423C;'>📉 Declining</span>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<span style='color: {PL_GOLD};'>➡️ Stable</span>", unsafe_allow_html=True)

                                    st.write(result['content'])
                                    st.caption(
                                        f"🎯 Season: {result.get('predicted_points', 0):.1f} pts | "
                                        f"Next GW: {result.get('next_game_predicted_points', 0):.2f} | "
                                        f"Opponent: {result.get('next_game_opponent', 'Unknown')} | "
                                        f"Difficulty: {result.get('next_game_difficulty', 'Unknown')}"
                                    )

                st.markdown("---")
                st.markdown(f"<h3 style='color: {PL_BLUE};'>💡 Example Queries</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    - Which players are predicted to score well?
                    - Who has the best form and sentiment?
                    - Tell me about high-scoring defenders
                    - Which budget players offer value?
                    """)
                with col2:
                    st.markdown("""
                    - Who should I captain this week?
                    - Players with injury concerns
                    - Best midfielders under £8M
                    - Which forwards are improving?
                    """)
    
    with tab5:
        st.markdown(f"<h2>ℹ️ About This Dashboard</h2>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {PL_BLUE}15 0%, {PL_GOLD}10 100%);
                    padding: 20px; border-radius: 10px; border-left: 4px solid {PL_BLUE};">
            <h3 style="color: {PL_BLUE}; margin-top: 0;">⚽ Fantasy Premier League Manager</h3>
            <p>Your AI-powered companion for Fantasy Premier League team selection using advanced machine learning and natural language analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            ### 🔧 Core Features

            - **📊 Player Analytics**: View comprehensive player statistics with interactive filters
            - **🔮 ML Predictions**: Ensemble models (Random Forest, LightGBM, XGBoost) predict player performance
            - **💬 Sentiment Analysis**: AI-driven news sentiment analysis using transformer models
            - **🔍 RAG Search**: Ask natural language questions about players and get instant insights
            - **📈 Value Analysis**: Find the best price-to-performance players
            - **⚡ Real-time Data**: Direct integration with official Fantasy PL API
            """)

        with col2:
            st.markdown(f"""
            ### 🎯 How It Works

            1. **Data Collection**: Live player data from FPL API + news from multiple sources
            2. **Sentiment Analysis**: NLP models analyze 50+ articles per player for sentiment
            3. **Feature Engineering**: Rolling averages, form trends, opponent strength, efficiency metrics
            4. **Ensemble Prediction**: 3 ML models vote on performance predictions
            5. **RAG System**: FAISS vector database enables semantic search across all data
            """)

        st.markdown("---")

        st.markdown(f"""
        ### 📊 Understanding the Data

        **Sentiment Score (0-100)**
        - 🔴 **0-40**: Negative sentiment (injuries, poor form, missed chances)
        - 🟡 **40-60**: Neutral sentiment (mixed news, routine updates)
        - 🟢 **60-100**: Positive sentiment (goal-scoring form, assists, clean sheets)

        *Note: Players need 50+ news articles to receive a sentiment score*

        ---

        ### 🤖 Prediction Models

        The dashboard uses an **ensemble approach** combining three powerful models:
        - **Random Forest**: Excellent for feature interactions and explainability
        - **LightGBM**: Fast training with superior handling of tree depth
        - **XGBoost**: Different regularization for ensemble diversity

        **Features used:**
        - Rolling window statistics (3, 5, 10 game averages)
        - Form trends and consistency metrics
        - Strength of Schedule (opponent difficulty)
        - Efficiency ratings (goals/minutes, assists/creativity)
        - Team composition and position strength
        - Temporal validation (TimeSeriesSplit - no data leakage!)
        """)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: {PL_BLUE}20; padding: 15px; border-radius: 8px;">
            <h4 style="color: {PL_BLUE};"> 📡 Data Sources</h4>
            • FPL Official API<br>
            • Google News RSS<br>
            • Multi-season statistics<br>
            • Team fixture data
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: {PL_GOLD}20; padding: 15px; border-radius: 8px;">
            <h4 style="color: {PL_BLUE};"> 🛠️ Technologies</h4>
            • Python, Streamlit<br>
            • scikit-learn, XGBoost<br>
            • Sentence Transformers<br>
            • FAISS Vector DB
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: #7CB34220; padding: 15px; border-radius: 8px;">
            <h4 style="color: {PL_BLUE};">✨ Unique Aspect</h4>
            • Kaggle March Madness<br>
            &nbsp;&nbsp;techniques adapted for FPL<br>
            • Uncertainty quantification<br>
            • Next-game predictions
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"""
        ### ⚖️ FPL Rules & Constraints

        The system understands all Fantasy Premier League rules:
        - **Budget**: £100M for 15-player squad
        - **Positions**: 2 GK, 5 DEF, 5 MID, 3 FWD
        - **Team Limits**: Max 3 players per club
        - **Transfers**: Limited per season (costs points)
        - **Captain**: Double points for selected player
        - **Bench Boost**: Double points for all bench players (limited use)

        ---

        ### ⚠️ Disclaimer

        This dashboard is a **decision support tool**, not financial or betting advice. Always conduct your own research before making fantasy football decisions. Past performance doesn't guarantee future results.

        **Built with ❤️ using cutting-edge AI & open-source technologies**
        """)



if __name__ == "__main__":
    main()
