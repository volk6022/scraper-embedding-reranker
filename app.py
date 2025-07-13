from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from contextlib import asynccontextmanager

from models import (
    ScrapeRequest,
    ScrapeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse
)

from scraper import Scraper
from embedding import Embedder
from reranker import Reranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await scraper.close()
    await embedder.close()
    await reranker.close()

app = FastAPI(lifespan=lifespan)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
scraper = Scraper()
embedder = Embedder()
reranker = Reranker()

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_endpoint(request: ScrapeRequest):
    return await scraper.scrape(
        request
    )

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_endpoint(request: EmbeddingRequest):
    return await embedder.get_embeddings(
        request
    )

@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    return await reranker.rerank(
        request
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1000)
