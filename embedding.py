from typing import List
from fastapi import HTTPException
from transformers import AutoModel, AutoTokenizer
import torch
import asyncio
from models import EmbeddingResponse, EmbeddingData, EmbeddingUsage, EmbeddingRequest

class Embedder:
    def __init__(self):
        self.model_name = "jinaai/jina-embeddings-v3"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    async def get_embeddings(self, embedding_request: EmbeddingRequest) -> EmbeddingResponse:
        """Generates embeddings for a list of texts using the loaded model."""
        model = embedding_request.model
        input_texts = embedding_request.input
        dimensions = embedding_request.dimensions

        try:
            encoded_input = self.tokenizer(input_texts, padding=True, return_tensors='pt').to(self.device)
            
            prompt_tokens = torch.sum(encoded_input['attention_mask']).item()
            total_tokens = prompt_tokens

            with torch.no_grad():
                model_output = self.model(**encoded_input, return_dict=True)
                sentence_embeddings = model_output[0][:, 0]
                # normalize embeddings
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings = sentence_embeddings.cpu().tolist()

            data = [
                EmbeddingData(
                    index=i,
                    embedding=embedding
                ) for i, embedding in enumerate(embeddings)
            ]

            return EmbeddingResponse(
                model=self.model_name,
                data=data,
                usage=EmbeddingUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=total_tokens
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during getting embedding: {str(e)}"
            )

    async def close(self):
        pass
