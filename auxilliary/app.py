import streamlit as st
from RAG.langchain_rag import agent

st.set_page_config(page_title="Fantasy Manager", page_icon=":soccer:", layout="wide")
st.title("Fantasy Manager :soccer:")
st.caption("Your personal assistant for Fantasy Premier League insights and predictions.")

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
        with st.spinner("Watch the tapes..."):
            response = ""
            for step in agent.stream(
                {'messages': [{"role": "user", "content": prompt}]},
                stream_mode = 'values'
            ):
                last = step['message'][-1]
                if last.type == 'ai' and not last.tools_calls:
                    response = last.content
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})