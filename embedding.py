from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from transformers import AutoModel, AutoTokenizer
import torch
from models import EmbeddingResponse, EmbeddingData, EmbeddingUsage, EmbeddingRequest
from config import args


class Embedder:
    def __init__(self):
        self.model_name = "jinaai/jina-embeddings-v3"
        self.tokenizer = None
        self.model = None
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

        if not args.lazy_load:
            self._load_model()

    def _load_model(self):
        if self.model is None:
            print(f"Loading Embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.eval()
            if not args.low_vram or not self.device.startswith("cuda"):
                self.model.to(self.device)

    def _sync_get_embeddings(
        self, embedding_request: EmbeddingRequest
    ) -> EmbeddingResponse:
        input_texts = embedding_request.input
        self._load_model()

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to(self.device)

        encoded_input = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        prompt_tokens = torch.sum(encoded_input["attention_mask"]).item()
        total_tokens = prompt_tokens

        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
            sentence_embeddings = model_output[0][:, 0]
            # normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
            embeddings = sentence_embeddings.cpu().tolist()

        if args.low_vram and self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

        data = [
            EmbeddingData(index=i, embedding=embedding)
            for i, embedding in enumerate(embeddings)
        ]

        return EmbeddingResponse(
            model=self.model_name,
            data=data,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens, total_tokens=total_tokens
            ),
        )

    async def get_embeddings(
        self, embedding_request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Generates embeddings for a list of texts using the loaded model."""
        async with self.lock:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor, self._sync_get_embeddings, embedding_request
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error during getting embedding: {str(e)}"
                )

    async def close(self):
        self.executor.shutdown()
