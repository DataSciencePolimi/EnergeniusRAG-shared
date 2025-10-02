"""UI module for the Energenius GURU application."""

import json
from io import StringIO

import streamlit as st

from orchestrator import LiveOrchestrator

# give title to the page
st.title("Energenius GURU")
st.subheader("A Graph-based RAG on Energy Efficiency")
st.write(
    """Trained on Italian and Swiss energy documents. \
         Ask anything you want on Energy Efficiency."""
)

# set the messages in the session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# create sidebar to adjust parameters
st.sidebar.title("Model Parameters")
provider = st.sidebar.selectbox("Provider", ["openai", "ollama"], index=1)

if provider == "ollama":
    model = st.sidebar.selectbox("Model", ["gpt-oss", "llama3.2", "mistral"], index=0)
    embedding = st.sidebar.selectbox(
        "Embedding", ["mxbai-embed-large", "nomic-embed-text"], index=0
    )
elif provider == "openai":
    model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "ollama"], index=1)
    embedding = st.sidebar.selectbox(
        "Embedding", ["text-embedding-3-small", "text-embedding-3-large"], index=0
    )
else:
    model = "None"
    embedding = "None"

language = st.sidebar.selectbox(
    "Language", ["English", "Italiano", "Español", "Română"], index=0
)
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.75, step=0.1
)
#region = st.sidebar.selectbox(
#    "User country", ["Italy", "Switzerland", "Other"], index=0
#)
user_type = st.sidebar.selectbox(
    "Socioeconomic status", ["Low income", "Medium income", "High income"], index=0
)
house_type = st.sidebar.selectbox(
    "House cluster", ["Apartment", "Detached House", "Mansion/Cottage"], index=0
)
use_knowledge_base = st.sidebar.toggle(
    "Use Knowledge Base", value=True, help="Use the knowledge base to answer questions."
)


# Orchestrator initialization
orchestrator = LiveOrchestrator(
    provider=provider,
    model=model,
    embedding=embedding,
    language=language,
    temperature=temperature,
    user_type=user_type,
    house_type=house_type,
    region="Italy", #region,
    use_knowledge=use_knowledge_base,
)

# Support functions
st.sidebar.button(
    "Clear chat",
    icon=":material/delete:",
    on_click=lambda: st.session_state.pop("messages", None),
)

st.sidebar.download_button(
    label="Download chat",
    help="Download the chat history as a JSON file.",
    icon=":material/download:",
    file_name="chat.json",
    mime="application/json",
    data=json.dumps(st.session_state["messages"], separators=(",", ": ")),
)
chat_upload = st.sidebar.file_uploader(
    label="Upload chat",
    help="Upload a chat file to continue from a conversation.",
    type=["json"],
)
if chat_upload is not None:
    # erase the current chat
    st.session_state["messages"] = []

    # To convert to a string based IO:
    stringio = StringIO(chat_upload.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

    # Convert into JSON
    try:
        CHAT_DATA = json.loads(string_data)
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        CHAT_DATA = None

    if CHAT_DATA is not None:
        # check if the file is a valid chat file
        if isinstance(CHAT_DATA, list):
            # check if the file is a valid chat file
            for message in CHAT_DATA:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    st.error("Invalid chat file format.")
                    break
            else:
                # add the messages to the session state
                st.session_state["messages"] = CHAT_DATA
                #st.success("Chat file uploaded successfully.")

                # Call to the orchestrator to load the messages
                orchestrator.load_past_messages(CHAT_DATA)

        else:
            st.error("Invalid chat file format.")

# update the interface with the previous messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# create the chat interface
if prompt := st.chat_input("Enter your query"):
    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Display waiting message
    #with st.spinner("Thinking..."):
    #    # Calling the orchestrator to get the response
    #    agent = orchestrator.user_message(prompt)
    #
    #    # Store and display the response
    #    st.session_state["messages"].append({"role": "assistant", "content": agent})
    #    st.chat_message("assistant").markdown(agent)

    # Streaming: create an empty placeholder for assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Show spinner while streaming response
        with st.spinner(""):
            for chunk in orchestrator.user_message(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response)# + "▌")  # Show typing cursor

        response_placeholder.markdown(full_response)  # Final cleanup

        # Save final assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
