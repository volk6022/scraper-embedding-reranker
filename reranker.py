from fastapi import HTTPException
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models import RerankResponse, RerankResult, RerankRequest


class Reranker:
    def __init__(self):
        self.model_name = "jinaai/jina-reranker-v2-base-multilingual"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    async def rerank(
        self,
        rerank_request: RerankRequest,
    ) -> RerankResponse:
        model = rerank_request.model
        query = rerank_request.query
        documents = rerank_request.documents
        top_n = rerank_request.top_n
        try:
            if not documents:
                return RerankResponse(results=[])

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
                model=self.model_name,
                results=results[:top_n],
                usage={"total_tokens": 0}
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during reranking: {str(e)}"
            )

    async def close(self):
        pass
