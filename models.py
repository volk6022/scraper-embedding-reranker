from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple

# --- Reranker Models ---
class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: int
    documents: list[str]

class RerankResult(BaseModel):
    index: int
    document: dict
    relevance_score: float

class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]
    usage: dict

# --- Embedding Models ---
class EmbeddingRequest(BaseModel):
    model: str = "jina-embeddings-v3"
    texts: List[str]
    task: Optional[str] = "text-matching"
    truncate: Optional[bool] = True
    dimensions: Optional[int] = 128
    late_chunking: Optional[bool] = None
    embedding_type: Optional[str] = None

class EmbeddingData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str = "jina-embeddings-v3"
    object: str = "list"
    usage: EmbeddingUsage

# --- Scraper Models ---
class ImageData(BaseModel):
    url: str
    width: int
    height: int
    format: str
    size_bytes: int

class ScrapeRequest(BaseModel):
    url: str
    download_images: bool = False
    get_html: bool = False

class ScrapeResponse(BaseModel):
    data: dict
