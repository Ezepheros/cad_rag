import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from abc import ABC, abstractmethod
from cad_rag.utils.logging_util import get_logger
logger = get_logger(__name__)

class EmbeddingModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = None
        self.emb_model = None
        self.emb_dim = None
        self.index_ = None
        self.num_calls = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def embed(self, texts, **kwargs) -> list[list[float]]:
        self.num_calls += 1
        return self._embed(texts, **kwargs)
    
    def reset_call_count(self):
        self.num_calls = 0

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]:
        pass
        
    # def __call__(self, text: str) -> list[float]:
    #     raise NotImplementedError("Subclasses must implement this method.")

class SentenceEmbeddingModelWrapper(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.emb_model = None
        self.emb_dim = None
        self.index_ = None
        self.num_calls = 0
        
    def embed(self, texts, **kwargs) -> list[list[float]]:
        self.num_calls += 1
        return self._embed(texts, **kwargs)
    
    def reset_call_count(self):
        self.num_calls = 0

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]:
        pass


class DummyEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = "dummy-model", emb_dim: int = 1024):
        super().__init__(model_name)
        self.emb_dim = emb_dim
        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = [np.ones(self.emb_dim) for _ in texts]
        return embeddings
    
    def tokenizer(self):
        class DummyTokens:
            def __init__(self, text: str):
                self.tokens = text.split()
                self.ids = list(range(len(text.split())))
            
            def __iter__(self):
                return iter(self.tokens)
            
        class DummyTokenizer:
            def encode(self, text: str):
                return DummyTokens(text)
            
            def decode(self, tokens: list[int]):
                return " ".join([str(token) for token in tokens])
            
        return DummyTokenizer()

class VoyageaiEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, vo_client, model_name: str):
        import voyageai
        super().__init__(model_name)
        self.vo_client = vo_client
        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.vo_client.embed(texts, model=self.model_name).embeddings
        return embeddings

    def tokenizer(self):
        return self.vo_client.tokenizer(model=self.model_name)
    
class HFDocumentEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        from transformers import AutoTokenizer, AutoModel
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emb_model = AutoModel.from_pretrained(model_name)
        self.emb_dim = self.emb_model.config.hidden_size
        self.max_length = self.tokenizer.model_max_length
        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        # embed text
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
        return embeddings

class HFSentenceEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        super().__init__(model_name)
        self.emb_model = SentenceTransformer(model_name)
        self.emb_dim = self.emb_model.get_sentence_embedding_dimension()
        
    def _embed(self, texts: list[str], batch_size=32) -> list[list[float]]:
        self.emb_model.to(self.device)

        embeddings = self.emb_model.encode(texts, batch_size=batch_size, convert_to_numpy=True).tolist()
        return embeddings

class DDPHFSentenceEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        import torch.distributed as dist

        super().__init__(model_name)
        assert "cuda" in self.device or "gpu" in self.device # "Must use GPUs for distributed models"

        dist.init_process_group(backend="nccl")
        self.local_rank = local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        torch.cuda.set_device(self.local_rank)
        self.world_size = dist.get_world_size()

        self.emb_model = SentenceTransformer(model_name).to(self.local_rank)
        self.emb_dim = self.emb_model.get_sentence_embedding_dimension()
        
    def _embed(self, texts: list[str], batch_size=64) -> list[list[float]]:
        breakpoint()
        texts_per_gpu = texts[self.local_rank::self.world_size]

        embeddings = self.emb_model.encode(texts, batch_size=batch_size, device=f"cuda:self.local_rank", convert_to_numpy=True).tolist()
        return embeddings
    
if __name__ == "__main__":
    def testHFEmbeddingModelWrapper():
        model = HFSentenceEmbeddingModelWrapper(model_name="all-mpnet-base-v2")
        texts = ["Let me tell you about Bill.", "Bill is a software engineer. ", "Bill likes tigers and he doesn't like Lucy",
                 "This is another text about Alice", "Alice is a data scientist", "She loves cats and hates Bob.",
                 "The United States of America is a country located in North America.", "It is made up of 50 states, a federal district, five major self-governing territories, and various possessions.", "It's quite wild right now"]
        embeddings = model.embed(texts)
        print(f"Embeddings: {embeddings}")
        print(f"Embedding dimension: {model.emb_dim}")
        emb_np = np.array(embeddings)
        for i in range(1, len(texts)):
            print(f"Sentence '{texts[i-1]}' <-> '{texts[i]}' similarity: {np.dot(emb_np[i-1], emb_np[i]) / (np.linalg.norm(emb_np[i-1]) * np.linalg.norm(emb_np[i]))}")
        print(f"Embeddings shape: {emb_np.shape}")
        breakpoint()
        
    testHFEmbeddingModelWrapper()
    