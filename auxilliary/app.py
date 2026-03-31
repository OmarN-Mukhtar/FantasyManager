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
            try:
                for step in agent.stream(
                    {'messages': [{"role": "user", "content": prompt}]},
                    stream_mode='values'
                ):
                    # Handle different output formats safely
                    if 'message' in step and step['message']:
                        messages = step['message']
                        if isinstance(messages, list) and len(messages) > 0:
                            last = messages[-1]
                            if hasattr(last, 'type') and last.type == 'ai':
                                response = getattr(last, 'content', '')
                    elif isinstance(step, dict) and 'content' in step:
                        response = step['content']
                
                if not response:
                    response = "I couldn't generate a response. Please try again."
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})