import os
import uuid
from tqdm import tqdm

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate



from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama3-70b-8192")


system = """You are provided with a chunk of text from a scientific article about insomnia, along with the title and the high-level summary (or abstract) of the article for additional context. Your task is to create a concise, informative, and detailed summary of the chunk, capturing its main points, explanations, and any significant findings. Utilize the title and high-level summary to frame the chunk in the broader context of the article, ensuring that the summary reflects the relevance and significance of the content.
Instructions:
- Base the summary primarily on the chunk of text, but use the title and high-level summary for better context.
- Include important details, findings, and explanations, but avoid unnecessary citations or excessive numbers unless critical.
- The summary should be detailed, clear, and informative, highlighting the key ideas without losing depth.
- Provide the summary in as much detail as possible.
- Your response should only be the summary and nothing else.

Chunk of Text:
{chunk_of_text}

Your Summary:
"""

prompt = ChatPromptTemplate.from_messages([
    ('system', system)
])

summary_chain = prompt | llm | StrOutputParser()



#Create chunks using RecursiveTextSplitter
def create_chunks_from_directory(dir):
    documents = []
    headers_to_split_on = [('#', 'Title'), ('##', 'High-level Summary')]
    markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap = 0)
    for path in os.listdir(dir):
        if path.endswith('high_level.md'):
            continue
        path = os.path.join(dir, path)
        main_content = TextLoader(path).load()[0]

        abstract_path = path[:-3] + '_high_level.md'
        abstract_content = TextLoader(abstract_path).load()[0].page_content
        md_header_splits = markdown_header_splitter.split_text(abstract_content)
        title, summary = md_header_splits[0].metadata['Title'], md_header_splits[0].metadata['High-level Summary']

        main_content.metadata['Title'] = title
        main_content.metadata['Summary'] = summary

        documents.append(main_content)

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata['doc_id'] = str(uuid.uuid4())

    return chunks

#Create summaries from chunks using LLM
def create_summaries_from_chunks(chunks, summary_chain):
    summaries = []
    for chunk in tqdm(chunks):
        title, summary, unique_id = chunk.metadata['Title'], chunk.metadata['Summary'], chunk.metadata['doc_id']
        res = title + '\n' + summary + '\n\n' + chunk.page_content
        chunk_summary = summary_chain.invoke(res)
        chunk_summary_document = Document(page_content=chunk_summary, metadata={"doc_id": unique_id, "title": title})
        summaries.append(chunk_summary_document)

    return summaries