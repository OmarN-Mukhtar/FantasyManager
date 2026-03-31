import streamlit as st
from RAG.langchain_rag import agent
from langchain.messages import AIMessage, HumanMessage

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
        with st.spinner("Watching the tapes..."):
            response = ""
            try:
                # Stream returns full state with 'messages' (plural)
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": prompt}]},
                    stream_mode="values"
                ):
                    # Each chunk contains the full state at that point
                    latest_message = chunk["messages"][-1]
                    
                    # Check message type and extract content
                    if isinstance(latest_message, AIMessage) and not latest_message.tool_calls:
                        response = latest_message.content
                    elif isinstance(latest_message, HumanMessage):
                        # Skip user messages, we already showed them
                        pass
                
                if not response:
                    response = "I couldn't generate a response. Please try again."
            except Exception as e:
                response = f"Error: {str(e)}"
                st.error(response)

        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})