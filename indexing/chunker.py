import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cad_rag.data.Dataset import CanadianCaseLawDocumentDatabase
from cad_rag.embedding_models.embedding_models_wrapper import EmbeddingModelWrapper
from cad_rag.indexing.splitter import SentenceSplitter
import re
# from nltk.tokenize import sent_tokenize
import nltk.data
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class Chunker:
    def __init__(self):
        pass
    
class NaiveChunker(Chunker):
    """
        Naively chunks documents into fixed-size chunks without considering semantic boundaries.
    """
    
    def __init__(self, tokenizer, chunk_size: int = 512, overlap: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        
    def chunk(self, text, text_metadata: dict = {}) -> list[str]:
        # hardcoded for voyageai tokenizer right now
        tokens = self.tokenizer.encode(text).ids
        
        chunks = []
        start = 0
        num_tokens = len(tokens)
        end = num_tokens
        
        while start < num_tokens:
            end = min(start + self.chunk_size, num_tokens)
            chunk = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk)
            final_chunk = {'unofficial_text_en': chunk_text, **text_metadata}
            
            start = start + self.chunk_size - self.overlap
            chunks.append(final_chunk)
            
        return chunks
    
    def batch_chunk(self, texts: list[str], text_metadatas: list[dict] = []) -> list[list[str]]:
        all_chunks = []
        for i, text in enumerate(texts):
            metadata = text_metadatas[i] if i < len(text_metadatas) else {}
            chunks = self.chunk(text, text_metadata=metadata)
            all_chunks.append(chunks)
        return all_chunks
    
class TopicChunker(Chunker):
    """
        Chunks documents based on topic boundaries.
    """
    
    def __init__(self, tokenizer, topic_model, max_chunk_size: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_chunk_size = max_chunk_size
        self.topic_model = topic_model
    
    def chunk(self, document, sim_threshold=0.75, min_sentences=3):
        """
        Splits a document into semantic chunks based on topic shifts.

        Parameters:
        - document: str, the full text to split
        - model_name: str, embedding model from sentence-transformers
        - sim_threshold: float, cosine similarity threshold to detect topic change
        - min_sentences: int, number of sentences per initial unit

        Returns:
        - chunks: list of strings
        """
        # Load embedding model
        breakpoint()
        
        # Step 1: Split document into sentences
        sentences = nltk.tokenize.sent_tokenize(document)
        
        # Step 2: Group sentences into units (like mini-paragraphs)
        units = []
        for i in range(0, len(sentences), min_sentences):
            units.append(" ".join(sentences[i:i+min_sentences]))
        
        # Step 3: Compute embeddings
        embeddings = model.encode(units)
        
        # Step 4: Detect topic shifts
        chunks = []
        current_chunk = [units[0]]
        
        for i in range(1, len(units)):
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            if sim < sim_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [units[i]]
            else:
                current_chunk.append(units[i])
        
        # Add last chunk
        chunks.append(" ".join(current_chunk))
        
        return chunks
    
class TopicChunker(Chunker):
    
    def __init__(self, threshold: float = 0.6, max_chunk_character_size: int = 512, sentence_emb_model=None):
        super().__init__()
        
    def chunk(self, texts: str) -> list[str]:
        # Assumes text is split into sentences some way
        
        # embed each sentence
        
        # compute cosine similarity between adjacent sentences and merge sentences until similarity drops below threshold or max chunk size is reached
        
    
if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = model.tokenizer
    text = "This is a sample document. It contains multiple sentences. The purpose is to test the TopicChunker. We want to see how well it can chunk text based on topic boundaries. Each chunk should ideally not exceed the maximum chunk size specified.\
        Let's add more sentences to ensure we have enough content to create multiple chunks. This will help us evaluate the chunking process effectively.\
        Finally, we will print out the resulting chunks to verify their sizes and contents.\
        Here is another sentence to make sure we have enough data. The quick brown fox jumps over the lazy dog. Natural language processing is a fascinating field of study.\
        Chunking text is an important step in many NLP applications."
        
    breakpoint()
    chunker = TopicChunker(tokenizer, topic_model=model, max_chunk_size=50)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk: {chunk}\n")
        # print(f"Chunk {i+1}: {chunk['unofficial_text_en']}\n")
    print(f"Total Chunks: {len(chunks)}")
    breakpoint()
    
    