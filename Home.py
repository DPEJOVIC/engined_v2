# Test

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import openai


st.title("Chat with ENGINED")


def setup():

    # Set up OpenAI API client
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Select OpenAI model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o-mini"

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "Hi, I'm ENGINED. Ask me about Factorio and the problems you need to solve."}]

    # Initialise system prompt
    with open("sysprompt.txt", "r") as file:
        system_prompt = file.read()
    
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = system_prompt


setup()


Settings.llm = OpenAI(model=st.session_state["openai_model"], temperature=0.5, system_prompt=st.session_state["system_prompt"])


@st.cache_resource(show_spinner=False)
def load_data(): 
    with st.spinner(text="Loading knowledge base"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index
    

index = load_data()


# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
     st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)


# Chat logic
if prompt := st.chat_input("Ask questions here"):
     st.session_state["chat_history"].append({"role": "user", "content": prompt})


# Display the prior chat messages
for message in st.session_state.chat_history:
     with st.chat_message(message["role"]):
          st.markdown(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.chat_history[-1]["role"] != "assistant":
     with st.chat_message("assistant"):
          with st.spinner("Thinking..."):
               # Use the chat engine to generate a response
               response = st.session_state.chat_engine.chat(prompt)
               st.markdown(response.response)

               # Add the response to the chat history
               message = {"role": "assistant", "content": response.response}
               st.session_state.chat_history.append(message)