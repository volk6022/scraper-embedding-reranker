from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models import (
    ScrapeRequest,
    ScrapeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    ReaderRequest,
    ReaderResponse,
)

from scraper import Scraper
from embedding import Embedder
from reranker import Reranker
from reader import Reader
from config import args


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await scraper.close()
    await embedder.close()
    await reranker.close()
    await reader.close()


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
reader = Reader()


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_endpoint(request: ScrapeRequest):
    return await scraper.scrape(request)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embed_endpoint(request: EmbeddingRequest):
    return await embedder.get_embeddings(request)


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    return await reranker.rerank(request)


@app.post("/v1/reader", response_model=ReaderResponse)
async def reader_endpoint(request: ReaderRequest):
    return await reader.read(request)


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server with args: {args}")
    uvicorn.run(app, host=args.host, port=args.port)
