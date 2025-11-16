import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path
import pyarrow.parquet as pq

import pandas as pd
import faiss
import numpy as np
from embedding_models.embedding_models_wrapper import EmbeddingModelWrapper, HFSentenceEmbeddingModelWrapper, HFDocumentEmbeddingModelWrapper, DDPHFSentenceEmbeddingModelWrapper

from cad_rag.utils.logging_util import get_logger
logger = get_logger(__name__)

class Retriever:
    def __init__(self, embedding_path: str | Path, file_type="pickle"):
        self.embedding_path = embedding_path
        if file_type == "pickle":
            self.load_embeddings_from_pickle()
        elif file_type == "parquet":
            self.load_embeddings_from_parquet()
    
    def load_embeddings_from_parquet(self):
        logger.info("before reading pq")
        self.data = pq.read_table(self.embedding_path) 
        logger.info("after reading pq")

    def load_embeddings_from_pickle(self):
        logger.info("before reading pickle")
        self.data = pd.read_pickle(self.embedding_path)
        logger.info("after reading pickle")

    def build_faiss_index(self):
        # TODO only works for pickle
        emb = np.vstack(self.data["embedding"].to_numpy()).astype("float32")
        self.index = faiss.IndexFlatL2(emb.shape[1])
        self.index.add(emb)

    def _query_index(self, query_emb, k=5):
        distances, ids = self.index.search(query_emb[None, :], k)
        return (distances, ids)
    
    def query_data(self, query_emb, k=5):
        _, ids = self._query_index(query_emb, k=k)
        return self.data.iloc[ids[0]]

if __name__ == "__main__":
    def test_retriever():
        
        path = "/ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/cad_rag/runs/embedded_naive_chunked_ONCA_512-64_QWEN8B_almostDONE/ONCA_embedded_chunked.pkl"
        logger.info("Reading embeddings...")
        retriever = Retriever(path)

        logger.info("building index...")
        retriever.build_faiss_index()

        model = HFSentenceEmbeddingModelWrapper("Qwen/Qwen3-Embedding-8B")
        query = "If I bought a machine and want to get a refund because it was damaged upon delivery, can I sue if I am not refunded?"
        query_emb = model.embed(query)
        query_emb = np.array(query_emb)
        logger.info("querying index...")
        
        entries = retriever.query_data(query_emb, k=5)

        save_path = os.path.dirname(__file__)
        with open(os.path.join(save_path, "retrieval.txt"), "w") as f:
            f.write(f"query: {query}\n")

            for i, text in enumerate(entries['unofficial_text']):
                f.write(f"doc {i}: {text}\n -----------------------------------------------------------------\n")

    test_retriever()
        


