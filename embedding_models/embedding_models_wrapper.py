import voyageai
import numpy as np

class EmbeddingModelWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = None
        self.emb_model = None
        self.emb_dim = None
        self.index_ = None
        self.num_calls = 0
        
    def embed(self, texts, **kwargs) -> list[list[float]]:
        self.num_calls += 1
        return self._embed(texts, **kwargs)
    
    def reset_call_count(self):
        self.num_calls = 0
    
    def _embed():
        raise NotImplementedError("Subclasses must implement this method.")
        
    # def __call__(self, text: str) -> list[float]:
    #     raise NotImplementedError("Subclasses must implement this method.")
    
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
        super().__init__(model_name)
        self.vo_client = vo_client
        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.vo_client.embed(texts, model=self.model_name).embeddings
        return embeddings

    def tokenizer(self):
        return self.vo_client.tokenizer(model=self.model_name)
    