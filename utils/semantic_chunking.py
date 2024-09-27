import os
import uuid
from tqdm import tqdm
from typing import List
import time

from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from semantic_encoder import BaseEncoder, BGEM3FlagEmbedEncoder
from semantic_chunkers.splitters.regex import RegexSplitter
from semantic_chunkers.schema import Chunk
from semantic_chunkers import StatisticalChunker

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.1-70b-versatile")


system = """Given the following chunk of text from a scientific article about insomnia, which includes the title and a high-level summary of the article, generate a detailed summary of the content. The summary should capture key information and provide enough detail to distinguish this chunk from others in the document. Focus only on the relevant information from this chunk, and avoid including excessive numbers or specific details from tables. Instead, capture the main points or conclusions derived from the data.
Instructions:
- Create a detailed summary of the chunk, utilizing the title and high-level summary for better context.
- Summarize the main points, specific findings, or arguments present in the chunk, but avoid excessive numeric data.
- Do not include any external information or analysis beyond what is provided in the chunk.
- Focus on distinguishing this chunk from other parts of the document by highlighting unique details.
- The output should contain only the detailed summary, nothing else."""

prompt = ChatPromptTemplate.from_messages([
    ('system', system),
    ('human', "Input:\n{chunk_of_text}")
])

summary_chain = prompt | llm | StrOutputParser()

def create_summaries_from_chunks(chunks, summary_chain):
    summaries = []
    for chunk in tqdm(chunks):
        title, summary, unique_id = chunk.metadata['Title'], chunk.metadata['Summary'], chunk.metadata['doc_id']
        res = title + '\n' + summary + '\n\nChunk of Text:\n' + chunk.page_content
        chunk_summary = summary_chain.invoke(res)
        chunk_summary_document = Document(page_content=chunk_summary, metadata={"doc_id": unique_id, "title": title})
        summaries.append(chunk_summary_document)

    return summaries

#Use after using a Chunker from semantic-chunkers to reformat the chunks (with overlap between chunks).
def reformat_semantic_chunks_with_overlap(original: Document, semantic_chunks: List[List[Chunk]], overlap=2):
    reformat_chunks = []
    level1 = RegexSplitter()(doc=original.page_content, delimiters=['\n\n'])
    level2 = RegexSplitter()(doc=original.page_content, delimiters=['\n'])
    for i, chunks in enumerate(semantic_chunks[0]):
        if overlap > 0:
            if i == 0:
                prechunks = []
            else:
                prechunks = semantic_chunks[0][i-1].splits
        else:
            prechunks = []

        chunks = chunks.splits
        res = ''

        if len(prechunks) > overlap:
            chunks = prechunks[-overlap:] + chunks
        else:
            chunks = prechunks + chunks

        for chunk in chunks:
            for paragraph in level1:
                if (chunk in paragraph):
                    if (chunk + '\n\n') in (paragraph + '\n\n'):
                        chunk = chunk + '\n\n'
                        break
                    else:
                        for paragraph_level2 in level2:
                            if chunk in paragraph_level2:
                                if (chunk + '\n') in (paragraph_level2 + '\n'):
                                    chunk = chunk + '\n'
                                else:
                                    chunk = chunk + ' '
            res += chunk
        reformat_chunks.append(Document(page_content=res, metadata=original.metadata))
    
    return reformat_chunks


#Create chunks using semantic chunker
def create_semantic_chunks_from_directory_with_overlap(dir: str, encoder: BaseEncoder, min_split_tokens=150, max_split_tokens=600, window_size=15, overlap=2):
    statistical_chunker = StatisticalChunker(encoder=encoder, min_split_tokens=min_split_tokens, max_split_tokens=max_split_tokens, window_size=window_size)
    documents = []
    headers_to_split_on = [('#', 'Title'), ('##', 'High-level Summary')]
    markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    for path in os.listdir(dir):
        if path.endswith('_high_level.md'):
            continue
        path = os.path.join(dir, path)
        main_content = TextLoader(path).load()[0]
        
        abstract_path = path[:-3] + '_high_level.md'
        abstract_content = TextLoader(abstract_path).load()[0].page_content
        title_summary = markdown_header_splitter.split_text(abstract_content)
        title, summary = title_summary[0].metadata['Title'], title_summary[0].metadata['High-level Summary']

        main_content.metadata['Title'] = title
        main_content.metadata['Summary'] = summary
        
        documents.append(main_content)
        
    final_semantic_chunks = []

    for doc in documents:
        semantic_chunks = statistical_chunker(docs = [doc.page_content])
        semantic_chunks = reformat_semantic_chunks_with_overlap(original=doc, semantic_chunks=semantic_chunks, overlap=overlap)
        
        for chunk in semantic_chunks:
            chunk.metadata['doc_id'] = str(uuid.uuid4())
            final_semantic_chunks.append(chunk)

            time.sleep(1.5)

    return final_semantic_chunks



#Create summaries from chunks using LLM
def create_summaries_from_chunks(chunks, summary_chain):
    summaries = []
    for chunk in tqdm(chunks):
        title, summary, unique_id = chunk.metadata['Title'], chunk.metadata['Summary'], chunk.metadata['doc_id']
        res = title + '\n' + summary + '\n\nChunk of Text:\n' + chunk.page_content
        chunk_summary = summary_chain.invoke(res)
        chunk_summary_document = Document(page_content=chunk_summary, metadata={"doc_id": unique_id, "title": title})
        summaries.append(chunk_summary_document)

    return summaries


"""
#Use after using a Chunker from semantic-chunkers to reformat the chunks.
def reformat_semantic_chunks(original: Document, semantic_chunks: List[List[Chunk]]):
    reformat_chunks = []
    level1 = RegexSplitter()(doc=original.page_content, delimiters=['\n\n'])
    level2 = RegexSplitter()(doc=original.page_content, delimiters=['\n'])
    for i, chunks in enumerate(semantic_chunks[0]):
        res = ''
        for chunk in chunks.splits:
            for paragraph in level1:
                if (chunk in paragraph):
                    if (chunk + '\n\n') in (paragraph + '\n\n'):
                        chunk = chunk + '\n\n'
                        break
                    else:
                        for paragraph_level2 in level2:
                            if chunk in paragraph_level2:
                                if (chunk + '\n') in (paragraph_level2 + '\n'):
                                    chunk = chunk + '\n'
                                else:
                                    chunk = chunk + ' '
            res += chunk
        reformat_chunks.append(Document(page_content=res, metadata=original.metadata))
    return reformat_chunks

#Create chunks using semantic chunker
def create_semantic_chunks_from_directory(dir: str, encoder: BaseEncoder, min_split_tokens=150, max_split_tokens=800, window_size=15):
    statistical_chunker = StatisticalChunker(encoder=encoder, min_split_tokens=min_split_tokens, max_split_tokens=max_split_tokens, window_size=window_size)
    documents = []
    headers_to_split_on = [('#', 'Title'), ('##', 'High-level Summary')]
    markdown_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

    for path in os.listdir(dir):
        if path.endswith('_high_level.md'):
            continue
        path = os.path.join(dir, path)
        main_content = TextLoader(path).load()[0]
        
        abstract_path = path[:-3] + '_high_level.md'
        abstract_content = TextLoader(abstract_path).load()[0].page_content
        title_summary = markdown_header_splitter.split_text(abstract_content)
        title, summary = title_summary[0].metadata['Title'], title_summary[0].metadata['High-level Summary']

        main_content.metadata['Title'] = title
        main_content.metadata['Summary'] = summary
        
        documents.append(main_content)
        
    final_semantic_chunks = []

    for doc in documents:
        semantic_chunks = statistical_chunker(docs = [doc.page_content])
        semantic_chunks = reformat_semantic_chunks(doc, semantic_chunks)
        
        for chunk in semantic_chunks:
            chunk.metadata['doc_id'] = str(uuid.uuid4())
            final_semantic_chunks.append(chunk)

    return final_semantic_chunks
"""