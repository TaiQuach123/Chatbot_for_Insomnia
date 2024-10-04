from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


system_reformulate_prompt = """Using the provided chat history (if available) and the most recent user question, reformulate the question into a standalone version that is understandable without the context of the chat history. Do not answer or provide any information beyond rephrasing. If rephrasing is unnecessary, return the original question exactly as it is. Your response must only be the reformulated or original question."""

reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system", system_reformulate_prompt),
    MessagesPlaceholder("messages")
])

system_prompt = """You are an AI chatbot designed to answer questions about insomnia using context retrieved from scientific articles. Your goal is to provide clear, evidence-based responses and practical advice to help users. Following those instructions:

1. Use only the information provided with the <context> tags regarding insomnia.
2. Provide concise, informative answers.
3. If a question is unclear or needs more context, ask the user for clarification.

<context>
{context}
</context>
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("messages"),
    ("human", "{query}")
])