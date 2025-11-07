import voyageai
import numpy as np
from sentence_transformers import SentenceTransformer

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
    
class HFSentenceEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        super().__init__(model_name)
        self.emb_model = SentenceTransformer(model_name)
        self.emb_dim = self.emb_model.get_sentence_embedding_dimension()
        
    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.emb_model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings
    
    def tokenizer(self):
        class SentenceTransformerTokenizer:
            def __init__(self, model):
                self.model = model
            
            def encode(self, text: str):
                tokens = self.model.tokenizer.tokenize(text)
                token_ids = self.model.tokenizer.convert_tokens_to_ids(tokens)
                return 
            
            def decode(self, token_ids: list[int]):
                tokens = self.model.tokenizer.convert_ids_to_tokens(token_ids)
                return self.model.tokenizer.convert_tokens_to_string(tokens)
        
        return SentenceTransformerTokenizer(self.emb_model)
    
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
    