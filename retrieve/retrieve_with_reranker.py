from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from utils.utils import convert_defaultdict
import numpy as np





def retrieve_with_reranker(query, embeddings, reranker, client):
    res = embeddings.encode([query], max_length=512, return_sparse=True, return_colbert_vecs=True)
    result = client.query_points(
        "summary",
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

    scores = reranker.compute_score([[query, point.payload['content']] for point in result.points], max_length=8096, batch_size=8, normalize=True)
    scores = np.array(scores)
    reranking_result = list(np.array(result.points)[scores.argsort()][::-1])


    relevant_docs = []
    for point in reranking_result[:5]:
        doc = client.scroll(
            collection_name="original",
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
