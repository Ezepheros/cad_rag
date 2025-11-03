import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cad_rag.data.Dataset import CanadianCaseLawDocumentDatabase
import re
from nltk.tokenize import sent_tokenize
import nltk.data
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path

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