import os  # Import the os module for interacting with the operating system
import tempfile  # Import the tempfile module for creating temporary files
import streamlit as st  # Import the Streamlit library for creating web apps
from streamlit_chat import message  # Import the message function from the streamlit_chat module
from rag import ChatPDF  # Import the ChatPDF class from the rag module

# Set the configuration for the Streamlit page
st.set_page_config(page_title="ChatPDF")

def display_messages():
    # Display a subheader for the chat section
    st.subheader("Chat")
    # Iterate over the messages stored in the session state and display each one
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    # Create an empty placeholder for the thinking spinner
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    # Check if the user input is not empty or just whitespace
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()  # Get the trimmed user input
        # Display a spinner while processing the input
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)  # Get the response from the assistant

        # Append the user input and assistant response to the messages list
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def page():
    # Initialize session state variables if they are not already set
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    # Display the main header for the app
    st.header("ChatPDF")

    # Display a subheader for the document upload section
    st.subheader("Upload a document")
    # Create a file uploader widget for uploading PDF documents
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Create an empty placeholder for the ingestion spinner
    st.session_state["ingestion_spinner"] = st.empty()

    # Display the chat messages
    display_messages()
    # Create a text input widget for user messages
    st.text_input("Message", key="user_input", on_change=process_input)

def read_and_save_file():
    # Clear the assistant's data and reset session state variables
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Iterate over the uploaded files
    for file in st.session_state["file_uploader"]:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())  # Write the file content to the temporary file
            file_path = tf.name  # Get the path of the temporary file

        # Display a spinner while ingesting the file
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)  # Ingest the file using the assistant
        os.remove(file_path)  # Remove the temporary file

# Run the page function if this script is executed directly
if __name__ == "__main__":
    page()