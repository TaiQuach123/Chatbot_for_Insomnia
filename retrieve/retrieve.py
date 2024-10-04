from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
from utils.utils import convert_defaultdict





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