from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.utils import convert_defaultdict, format_docs
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

if 'client' not in st.session_state:
    st.session_state.client = QdrantClient("http://localhost:6333")
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(model="llama3-70b-8192")


#system = """You are an AI chatbot designed to answer questions about insomnia using context retrieved from scientific articles. Your goal is to provide clear, evidence-based responses and practical advice to help users manage their insomnia. Following those instructions:
#1. Context: Use only the information provided with the <context> tags regarding insomnia. If possible, provide which chunk it from.
#2. Answer Format: Provide concise, informative answers.
#3. User Interaction: If a question is unclear or needs more context, ask the user for clarification.

#<context>
#{context}
#</context>"""

#human = """{query}"""

#prompt = ChatPromptTemplate.from_messages([
#    ("system", system),
#Q    ("human", human)
#])
prompt = ChatPromptTemplate.from_template("""Answer the question based on the provided context only. Try your best to provide the most accurate response.
<context>
{context}
</context>

Question: {query}
""")

chain = prompt | st.session_state.llm | StrOutputParser()



def retrieve(query, embeddings, client):
    res = embeddings.encode([query], return_sparse=True, return_colbert_vecs=True)
    result = client.query_points(
        "semantic_summary_vectorstore",
        prefetch=[
            models.Prefetch(
                query=res['dense_vecs'][0],
                using="dense",
                limit=20
            ),
            models.Prefetch(
                query=models.SparseVector(**convert_defaultdict(res['lexical_weights'][0])),
                using="sparse",
                limit=20
            ),
            models.Prefetch(
                query=res['colbert_vecs'][0],
                using='colbert',
                limit=20
            )
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=10
    )

    relevant_docs = []
    for point in result.points:
        doc = client.scroll(
            collection_name="semantic_original",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=point.id)
                    )
                ]
            )
        )

        temp_payload = doc[0][0].payload
        res_doc = Document(page_content=temp_payload['page_content'], metadata={'source':temp_payload['source'], 'doc_id': temp_payload['doc_id'], 'title': temp_payload['title']})
        relevant_docs.append(res_doc)
    
    
    return relevant_docs


st.title("Demo")
query = st.text_input("Enter Your Query")
if query:
    relevant_docs = retrieve(query, embeddings=st.session_state.embeddings, client=st.session_state.client)
    context = format_docs(relevant_docs[:5])

    response = chain.invoke({"context": context, "query": query})
    st.write(response)
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(relevant_docs):
            st.write(doc.metadata['source'] +  '|----------------|' + doc.metadata['title'] + '|----------------|' + doc.page_content)
            st.write('-'*100)