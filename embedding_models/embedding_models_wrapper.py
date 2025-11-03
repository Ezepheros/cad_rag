import voyageai

class EmbeddingModelWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider = None
        self.emb_model = None
        self.emb_dim = None
        self.index_ = None
        
    def embed():
        raise NotImplementedError("Subclasses must implement this method.")
        
    # def __call__(self, text: str) -> list[float]:
    #     raise NotImplementedError("Subclasses must implement this method.")
    

class VoyageaiEmbeddingModelWrapper(EmbeddingModelWrapper):
    def __init__(self, vo_client, model_name: str):
        super().__init__(model_name)
        self.vo_client = vo_client
        
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.vo_client.embed(texts, model=self.model_name).embeddings
        return embeddings

    def tokenizer(self):
        return self.vo_client.tokenizer(model=self.model_name)
    