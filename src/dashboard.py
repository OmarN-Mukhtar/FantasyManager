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
from rag_system import PlayerNewsRAG


# Page configuration
st.set_page_config(
    page_title="Fantasy PL Manager",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_player_data():
    """Load player data with sentiment scores."""
    try:
        df = pd.read_csv('data/players_with_sentiment.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please run data_fetcher.py and sentiment_analyzer.py first")
        return None


@st.cache_data
def load_sentiment_data():
    """Load detailed sentiment analysis."""
    try:
        with open('data/sentiment_analysis.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return []


@st.cache_data
def load_predictions_data():
    """Load prediction data."""
    try:
        with open('data/predictions.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}


@st.cache_resource
def load_rag_system():
    """Load RAG system with FAISS."""
    try:
        rag = PlayerNewsRAG()
        if rag.load_data():
            # Try to load existing index
            if not rag._load_index():
                st.warning("Vector database not found. Please run: python src/rag_system.py")
            return rag
        return None
    except Exception as e:
        st.warning(f"RAG system not available: {e}")
        return None


def main():
    """Main dashboard function."""
    
    # Header
    st.title("⚽ Fantasy Premier League Manager")
    st.markdown("AI-powered player analysis with sentiment scoring and news insights")
    
    # Load data
    df = load_player_data()
    
    if df is None or df.empty:
        st.stop()
    
    sentiment_data = load_sentiment_data()
    predictions_data = load_predictions_data()
    
    # Merge predictions into dataframe
    if predictions_data:
        pred_df = pd.DataFrame(predictions_data.values())
        df = df.merge(
            pred_df[['player_name', 'predicted_total_points', 'predicted_points_per_match', 'form_trend']],
            left_on='name',
            right_on='player_name',
            how='left'
        )
        df = df.drop('player_name', axis=1)
    else:
        df['predicted_total_points'] = 0
        df['predicted_points_per_match'] = 0
        df['form_trend'] = 'unknown'
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Team filter
    all_teams = sorted(df['team'].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=all_teams,
        default=all_teams
    )
    
    # Position filter
    all_positions = sorted(df['position'].unique())
    selected_positions = st.sidebar.multiselect(
        "Select Positions",
        options=all_positions,
        default=all_positions
    )
    
    # Sentiment filter
    st.sidebar.subheader("Sentiment Filter")
    show_only_scored = st.sidebar.checkbox("Show only players with sentiment scores", value=False)
    
    if not show_only_scored:
        min_articles = 0
    else:
        min_articles = st.sidebar.slider("Minimum articles", 0, 100, 50)
    
    # Price filter
    st.sidebar.subheader("Price Range")
    price_range = st.sidebar.slider(
        "Select price range (£M)",
        float(df['now_cost'].min()),
        float(df['now_cost'].max()),
        (float(df['now_cost'].min()), float(df['now_cost'].max())),
        0.1
    )
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Player Table", "🔮 Predictions", "📈 Analytics", "🔍 RAG Search", "ℹ️ About"])
    
    with tab1:
        st.header("Player Data")
        
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
                }[x]
            )
        with col2:
            ascending = st.checkbox("Ascending", value=False)
        
        # Sort dataframe
        display_df = filtered_df.sort_values(sort_by, ascending=ascending)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(display_df))
        with col2:
            scored_players = len(display_df[display_df['has_enough_articles'] == True])
            st.metric("With Sentiment Scores", scored_players)
        with col3:
            avg_sentiment = display_df['normalized_score'].mean()
            if pd.notna(avg_sentiment):
                st.metric("Avg Sentiment", f"{avg_sentiment:.1f}/100")
            else:
                st.metric("Avg Sentiment", "N/A")
        with col4:
            total_articles = display_df['article_count'].sum()
            st.metric("Total Articles", f"{int(total_articles):,}")
        
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
        st.header("🔮 Player Performance Predictions")
        
        if not predictions_data:
            st.warning("No predictions available. Please run: python src/predictor.py")
        else:
            # Sort options for predictions
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by_pred = st.selectbox(
                    "Sort predictions by",
                    ["predicted_total_points", "predicted_points_per_match", "now_cost", "total_points"],
                    format_func=lambda x: {
                        "predicted_total_points": "Predicted Total Points",
                        "predicted_points_per_match": "Predicted PPM",
                        "now_cost": "Price",
                        "total_points": "Current Points"
                    }[x]
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
                st.metric("Players with Predictions", len(pred_display_df))
            with col2:
                avg_pred = pred_display_df['predicted_total_points'].mean()
                st.metric("Avg Predicted Points", f"{avg_pred:.1f}")
            with col3:
                top_pred = pred_display_df['predicted_total_points'].max()
                st.metric("Highest Prediction", f"{top_pred:.1f}")
            with col4:
                improving = len(pred_display_df[pred_display_df['form_trend'] == 'improving'])
                st.metric("Players Improving", improving)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 20 Predicted Points")
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
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Predicted Points by Position")
                fig = px.box(
                    pred_display_df,
                    x='position',
                    y='predicted_total_points',
                    color='position',
                    title="Point Distribution by Position"
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Value analysis
            st.subheader("Value Analysis: Price vs Predicted Points")
            fig = px.scatter(
                pred_display_df,
                x='now_cost',
                y='predicted_total_points',
                color='position',
                size='total_points',
                hover_data=['name', 'team', 'form_trend'],
                title="Price vs Predicted Performance (size = current points)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Predictions table
            st.subheader("Player Predictions")
            pred_columns = [
                'name', 'team', 'position', 'now_cost', 'predicted_total_points',
                'predicted_points_per_match', 'form_trend', 'total_points'
            ]
            
            pred_table = pred_display_df[pred_columns].copy()
            pred_table.columns = [
                'Player', 'Team', 'Pos', 'Price', 'Pred Total', 'Pred PPM',
                'Form', 'Current Pts'
            ]
            
            # Format numbers
            pred_table['Pred Total'] = pred_table['Pred Total'].apply(lambda x: f"{x:.1f}")
            pred_table['Pred PPM'] = pred_table['Pred PPM'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                pred_table,
                use_container_width=True,
                height=400,
                hide_index=True
            )
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Only show analytics for players with sentiment scores
        scored_df = filtered_df[filtered_df['has_enough_articles'] == True].copy()
        
        if len(scored_df) == 0:
            st.warning("No players with sentiment scores in current filter. Adjust filters to see analytics.")
        else:
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution by Team")
                fig = px.box(
                    scored_df,
                    x='team',
                    y='normalized_score',
                    color='team',
                    title="Sentiment Scores by Team"
                )
                fig.update_xaxis(tickangle=45)
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Sentiment vs Points")
                fig = px.scatter(
                    scored_df,
                    x='normalized_score',
                    y='total_points',
                    color='position',
                    size='now_cost',
                    hover_data=['name', 'team'],
                    title="Sentiment Score vs Total Points"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Top performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Most Positive Sentiment")
                top_positive = scored_df.nlargest(10, 'normalized_score')[
                    ['name', 'team', 'normalized_score', 'total_points']
                ]
                for idx, row in top_positive.iterrows():
                    st.write(f"**{row['name']}** ({row['team']}) - {row['normalized_score']:.1f}/100")
            
            with col2:
                st.subheader("📉 Most Negative Sentiment")
                top_negative = scored_df.nsmallest(10, 'normalized_score')[
                    ['name', 'team', 'normalized_score', 'total_points']
                ]
                for idx, row in top_negative.iterrows():
                    st.write(f"**{row['name']}** ({row['team']}) - {row['normalized_score']:.1f}/100")
            
            # Sentiment by position
            st.subheader("Average Sentiment by Position")
            position_sentiment = scored_df.groupby('position')['normalized_score'].mean().reset_index()
            fig = px.bar(
                position_sentiment,
                x='position',
                y='normalized_score',
                title="Average Sentiment Score by Position",
                color='normalized_score',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price vs Sentiment
            st.subheader("Value Analysis: Price vs Sentiment")
            fig = px.scatter(
                scored_df,
                x='now_cost',
                y='normalized_score',
                color='position',
                size='total_points',
                hover_data=['name', 'team'],
                title="Price vs Sentiment (size = total points)"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔍 RAG Search - Ask Questions About Players")
        
        # Load RAG system
        rag = load_rag_system()
        
        if rag is None:
            st.warning("RAG system not available. Please run rag_system.py first to create the vector database.")
        else:
            st.markdown("Search through player news articles using natural language queries.")
            
            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., Which players are in good form? Who has injury concerns?"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                player_filter = st.selectbox(
                    "Filter by player (optional)",
                    ["All players"] + sorted(df['name'].unique().tolist())
                )
            with col2:
                doc_type = st.selectbox(
                    "Document type",
                    ["All", "News", "Predictions"]
                )
            with col3:
                n_results = st.slider("Number of results", 1, 20, 5)
            
            if query:
                with st.spinner("Searching..."):
                    # Execute query
                    filter_player = None if player_filter == "All players" else player_filter
                    filter_type = None if doc_type == "All" else doc_type.lower()[:-1]  # Remove 's'
                    results = rag.query(
                        query, 
                        n_results=n_results, 
                        player_filter=filter_player,
                        doc_type_filter=filter_type
                    )
                    
                    st.success(f"Found {len(results)} relevant results")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        result_type = result.get('type', 'news')
                        
                        if result_type == 'news':
                            with st.expander(
                                f"#{i} - {result['player_name']} - {result['title'][:60]}...", 
                                expanded=i==1
                            ):
                                st.markdown(f"**{result['player_name']}** ({result['team']})")
                                st.markdown(f"**{result['title']}**")
                                st.caption(f"Source: {result.get('source', 'Unknown')} | "
                                         f"Published: {result.get('published', 'Unknown')}")
                                st.markdown(f"**Sentiment Score:** {result.get('sentiment_score', 0):.1f}/100")
                                st.write(result['content'][:800] + "...")
                                if result.get('link'):
                                    st.markdown(f"[Read full article]({result['link']})")
                        else:
                            with st.expander(
                                f"#{i} - {result['player_name']} - Prediction",
                                expanded=i==1
                            ):
                                st.markdown(f"**{result['player_name']}** ({result['team']})")
                                st.markdown(f"**Performance Forecast**")
                                st.write(result['content'])
                                st.caption(f"Form Trend: {result.get('form_trend', 'unknown').title()}")
            
            # Example queries
            st.markdown("---")
            st.subheader("💡 Example Queries")
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
        st.header("About This Dashboard")
        
        st.markdown("""
        ### Fantasy Premier League Manager
        
        This dashboard provides AI-powered analysis of Premier League players using:
        
        1. **Data Fetching**: Real-time player data from the official Fantasy Premier League API
        2. **News Aggregation**: Collects recent news articles for each player from Google News
        3. **Sentiment Analysis**: Analyzes news sentiment to generate normalized scores (0-100)
        4. **Performance Prediction**: ML models predict player points using rolling window features
        5. **Interactive Dashboard**: Sort and filter players by multiple metrics
        6. **RAG System**: FPL-aware natural language search using FAISS vector database
        
        #### Features:
        
        - **Player Table**: View all players with sentiment scores, sortable by multiple criteria
        - **Predictions**: ML-powered predictions of player performance for the season
        - **Analytics**: Visualize sentiment distributions, correlations, and insights
        - **RAG Search**: Natural language search with FPL rules knowledge
        - **Filters**: Filter by team, position, price, and sentiment criteria
        
        #### Sentiment Score:
        
        The sentiment score (0-100) is calculated from news articles:
        - **0-40**: Negative sentiment (poor performance, injuries, concerns)
        - **40-60**: Neutral sentiment (mixed or routine news)
        - **60-100**: Positive sentiment (good form, goals, assists)
        
        Players need at least 50 news articles to receive a sentiment score.
        
        #### Prediction Model:
        
        Predictions use Random Forest models trained on historical data with:
        - Rolling window features (3, 5, 10 game averages)
        - Form trends and consistency metrics
        - Position and fixture analysis
        - All free - no API costs!
        
        #### Data Sources:
        
        - Player data: [Fantasy Premier League API](https://fantasy.premierleague.com/api/bootstrap-static/)
        - Historical data: Multi-season player statistics
        - News: Google News RSS feeds
        - Sentiment: TextBlob NLP library
        - Vector DB: FAISS (free, local)
        - ML Models: scikit-learn Random Forest
        
        #### FPL Rules:
        
        The RAG system understands FPL team selection rules including:
        - £100M budget, 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD)
        - Max 3 players per team
        - Starting XI formation requirements
        - Captain/Vice-Captain scoring
        - Points system by position
        
        ---
        
        **Note**: This tool is for informational purposes. Always do your own research before making fantasy football decisions!
        """)


if __name__ == "__main__":
    main()
