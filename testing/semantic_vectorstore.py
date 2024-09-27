from utils.semantic_chunking import create_semantic_chunks_from_directory_with_overlap, reformat_semantic_chunks_with_overlap
from utils.utils import batch_iterator, convert_defaultdict
from semantic_encoder import BGEM3FlagEmbedEncoder
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models
import argparse
import torch

client = QdrantClient("http://localhost:6333")

#Creating vectorstore (if not exist)
if not client.collection_exists(collection_name="semantic_vectorstore"):
    client.create_collection(
        "semantic_vectorstore",
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            ),
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

#embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

parser = argparse.ArgumentParser(description="Qdrant Vectorstore")
parser.add_argument('--dir', default="./extracted/TÁC HẠI", help="directory to create -> adding chunks to vectorstore")

args = parser.parse_args()

if __name__ == "__main__":
    embeddings = BGEM3FlagEmbedEncoder()
    chunks = create_semantic_chunks_from_directory_with_overlap(dir=args.dir, encoder=embeddings, overlap=2)
    embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    torch.cuda.empty_cache()
    
    batch_size = 8
    for batch in batch_iterator(chunks, batch_size):
        text = [(chunk.metadata['Title'] + '\n' + chunk.metadata['Summary'] + '\n\n' + 'Chunk of Text:\n' + chunk.page_content) for chunk in batch]
        res = embeddings.encode(text, return_sparse=True, return_colbert_vecs=True)

        for i, _ in enumerate(batch):
            doc_id = batch[i].metadata['doc_id']
            title = batch[i].metadata['Title']
            summary = batch[i].metadata['Summary']
            source = batch[i].metadata['source']
            content = batch[i].page_content
            try:
                client.upload_points(
                    "semantic_vectorstore",
                    points = [
                        models.PointStruct(
                            id = doc_id,
                            vector = {
                                "dense": res['dense_vecs'][i].tolist(),
                                "colbert": res['colbert_vecs'][i].tolist(),
                                "sparse": convert_defaultdict(res['lexical_weights'][i])
                            },
                            payload = {
                                "doc_id": doc_id,
                                "title": title,
                                "summary": summary,
                                "source": source,
                                "content": content,
                            }

                        )
                    ],
                    batch_size=1
                )
            except:
                print(f"Error when uploading - {doc_id}")
                continue
