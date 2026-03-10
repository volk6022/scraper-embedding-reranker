import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models import RerankResponse, RerankResult, RerankRequest
from config import args


class Reranker:
    def __init__(self):
        self.model_name = "jinaai/jina-reranker-v2-base-multilingual"
        self.tokenizer = None
        self.model = None
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

        if not args.lazy_load:
            self._load_model()

    def _load_model(self):
        if self.model is None:
            print(f"Loading Reranker model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.eval()
            if not args.low_vram or not self.device.startswith("cuda"):
                self.model.to(self.device)

    def _sync_rerank(self, rerank_request: RerankRequest) -> RerankResponse:
        query = rerank_request.query
        documents = rerank_request.documents
        top_n = rerank_request.top_n

        self._load_model()

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to(self.device)

        if not documents:
            return RerankResponse(
                results=[], model=self.model_name, usage={"total_tokens": 0}
            )

        if top_n is None:
            top_n = len(documents)

        features = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        features = {k: v.to(self.device) for k, v in features.items()}

        with torch.no_grad():
            scores = self.model(**features).logits.flatten().float().tolist()

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

        results = []
        for i, score in enumerate(scores):
            results.append(
                RerankResult(
                    index=i,
                    document={"text": documents[i]},
                    relevance_score=score,
                )
            )

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return RerankResponse(
            model=self.model_name, results=results[:top_n], usage={"total_tokens": 0}
        )

    async def rerank(self, rerank_request: RerankRequest) -> RerankResponse:
        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor, self._sync_rerank, rerank_request
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error during reranking: {str(e)}"
                )

    async def close(self):
        self.executor.shutdown()
