from typing import Any, Coroutine, List, Optional
from pydantic import BaseModel, Field, validator, PrivateAttr
import numpy as np
from FlagEmbedding import BGEM3FlagModel


#BaseEncoder for semantic-chunker
class BaseEncoder(BaseModel):
    name: str
    score_threshold: Optional[float] = None
    type: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

    @validator("score_threshold", pre=True, always=True)
    def set_score_threshold(cls, v):
        return float(v) if v is not None else None

    def __call__(self, docs: List[Any]) -> List[List[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    def acall(self, docs: List[Any]) -> Coroutine[Any, Any, List[List[float]]]:
        raise NotImplementedError("Subclasses must implement this method")
    

class BGEM3FlagEmbedEncoder(BaseEncoder):
    type: str = "FlagEmbed"
    name: str = "BAAI/bge-m3"
    max_length: int = 8096
    cache_dir: Optional[str] = None
    threads: Optional[int] = None
    _client: Any = PrivateAttr()
    def __init__(
        self, score_threshold: float = 0.5, **data
    ):  # TODO default score_threshold not thoroughly tested, should optimize
        super().__init__(score_threshold=score_threshold, **data)
        self._client = self._initialize_client()

    def _initialize_client(self):
        #embedding_args = {
        #    "model_name": self.name,
        #    "max_length": self.max_length,
        #    "cache_dir": self.cache_dir,
        #    "threads": self.threads,
        #}
        #embedding_args = {k: v for k, v in embedding_args.items() if v is not None}
        embedding = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        return embedding

    def __call__(self, docs: List[str]) -> List[List[float]]:
        try:
            embeds: List[np.ndarray] = list(self._client.encode(docs)['dense_vecs'])
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(f"FlagEmbed embed failed. Error: {e}") from e