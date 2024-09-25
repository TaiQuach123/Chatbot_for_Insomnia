import argparse
from utils.chunking import create_chunks_from_directory, create_summaries_from_chunks
from utils.chunking import summary_chain
from utils.utils import batch_iterator, convert_defaultdict
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel


client = QdrantClient("http://localhost:6333")

#Creating vectorstore (if not exist)
if not client.collection_exists(collection_name="summary"):
    client.create_collection(
        "summary",
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

if not client.collection_exists(collection_name="original"):
    client.create_collection("original", vectors_config={})


embeddings = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

parser = argparse.ArgumentParser(description="Qdrant Vectorstore")
parser.add_argument('--dir', default="./extracted/TÁC HẠI", help="directory to create -> adding chunks to vectorstore")

args = parser.parse_args()

if __name__ == "__main__":
    chunks = create_chunks_from_directory(args.dir)
    summaries = create_summaries_from_chunks(chunks, summary_chain=summary_chain)
    
    
    batch_size = 8
    for batch in batch_iterator(summaries, batch_size):
        text = [summary.page_content for summary in batch]
        res = embeddings.encode(text, return_sparse=True, return_colbert_vecs=True)

        for i, _ in enumerate(batch):
            doc_id = batch[i].metadata['doc_id']
            title = batch[i].metadata['title']
            content = batch[i].page_content
            try:
                client.upload_points(
                    "summary",
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
                                "content": content,
                            }

                        )
                    ],
                    batch_size=1
                )
            except:
                print(f"Error when uploading - {doc_id}")
                continue
    

    
    for chunk in chunks:
        id = chunk.metadata['doc_id']
        try:
            client.upload_points(
                "original",
                points = [
                    models.PointStruct(
                        id = id,
                        vector = {},
                        payload = {
                            "doc_id": id,
                            "source": chunk.metadata['source'],
                            "title": chunk.metadata['Title'],
                            "page_content": chunk.page_content,
                        }
                    )
                ],
                batch_size=1
            )
        except:
            print(f"Error when uploading - {id}")
            continue