import streamlit as st
import pandas as pd
from RAG.langchain_rag import agent

st.set_page_config(page_title="Fantasy Manager", page_icon=":soccer:", layout="wide")
st.title("Fantasy Manager :soccer:")
st.caption("Your personal assistant for Fantasy Premier League insights and predictions.")

tab_chat, tab_players = st.tabs(["💬 Chat", "📋 Players"])

with tab_chat:
    #Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input("Ask about player predictions, news, or FPL advice!"):
        #Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        #Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Watching the tapes..."):
                try:
                    chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ]
                    response = agent.invoke({"messages": chat_history})["messages"][-1].content
                    if not response:
                        response = "I couldn't generate a response. Please try again."
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)

            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab_players:
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
