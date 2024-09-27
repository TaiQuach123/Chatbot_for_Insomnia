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

llm = ChatGroq(model="llama-3.1-70b-versatile")

system = """Given the following chunk of text from a scientific article about insomnia, which includes the title and a high level summary of the article, generate a detailed summary of the content. The summary should capture key information and provide enough detail to distinguish this chunk from others in the document. Avoid including excessive numbers or specific details from tables. Instead, capture the main points or conclusions derived from the data.
Instructions:
- Create a detailed summary of the chunk, utilizing the title and high-level summary for better context.
- Summarize the main points, specific findings, or arguments present in the chunk, but avoid excessive numeric data.
- Do not include any external information or analysis beyond what is provided in the chunk.
- Focus on distinguishing this chunk from other parts of the document by highlighting unique details.
- The output should contain only the detailed summary, nothing else.

Input:
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