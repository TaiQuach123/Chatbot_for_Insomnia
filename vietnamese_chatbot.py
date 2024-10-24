from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.utils import convert_defaultdict, format_docs
import streamlit as st
from retrieve import retrieve
from prompts.vietnamese_promts import reformulate_prompt, prompt

from dotenv import load_dotenv
load_dotenv()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

if 'client' not in st.session_state:
    st.session_state.client = QdrantClient("http://localhost:6333")
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
if 'store' not in st.session_state:
    st.session_state.store = {}

session_id = "first_chat"
config = {"configurable": {"session_id": session_id}}



reformulate_chain = reformulate_prompt | st.session_state.llm | StrOutputParser()
final_chain = prompt | st.session_state.llm | StrOutputParser()
final_chain_with_memory = RunnableWithMessageHistory(final_chain, get_session_history, input_messages_key="query", history_messages_key="messages")


st.title("Demo Chatbot")
query = st.text_input("Enter Your Query")
if query:
    if not st.session_state.store.get(session_id, []):
        history = []
    else:
        history = st.session_state.store[session_id].messages
    
    reformulate_query = reformulate_chain.invoke({"messages": history, "query": query})
    print("Reformulate Query:")
    print(reformulate_query)
    print('-'*50)

    relevant_docs = retrieve(reformulate_query, embeddings=st.session_state.embeddings, client=st.session_state.client)
    context = format_docs(relevant_docs[:5])

    response = final_chain_with_memory.invoke({"query": query, "context": context}, config=config)
    st.write(response)

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(relevant_docs):
            st.write(doc.metadata['source'] + '|----------|' + doc.metadata['title'] + '|----------|' + doc.page_content)
            st.write('-'*75)
            st.write('-'*75)

    print(st.session_state.store[session_id].messages)