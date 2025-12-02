import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cad_rag.data.Dataset import CanadianCaseLawDocumentDatabase
from cad_rag.embedding_models.embedding_models_wrapper import EmbeddingModelWrapper
from cad_rag.indexing.splitter import SentenceSplitter
from cad_rag.utils.logging_util import get_logger
logger = get_logger(__name__)
import re
# from nltk.tokenize import sent_tokenize
import nltk.data
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from cad_rag.embedding_models.embedding_models_wrapper import EmbeddingModelWrapper, HFSentenceEmbeddingModelWrapper, HFDocumentEmbeddingModelWrapper, DDPHFSentenceEmbeddingModelWrapper

class Chunker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def chunk(self, text: str, text_metadata: dict = {}) -> list[dict]:
        pass
    
    def batch_chunk(self, texts: list[str], text_metadatas: list[dict] = []):
        all_chunks = []
        if text_metadatas:
            assert len(texts) == len(text_metadatas) # needs texts and text metadetas to be same lengths

            for text, text_metadata in zip(texts, text_metadatas):
                chunks = self.chunk(text, text_metadata=text_metadata)
                all_chunks.append(chunks)
        else:
            for text in texts:
                chunks = self.chunk(text, text_metadata={})
                all_chunks.append(chunks)
        
        return all_chunks # list of dictionaries 
    
    
class NaiveChunker(Chunker):
    """
        Naively chunks documents into fixed-size chunks without considering semantic boundaries.
    """
    
    def __init__(self, max_chunk_character_size: int = 512, overlap: int = 64):
        super().__init__()
        self.max_chunk_character_size = max_chunk_character_size
        self.overlap = overlap
        
    def chunk(self, text, text_metadata: dict = {}) -> list[str]:
        text_chunks = []
        idx = 0
        while idx < len(text):
            text_chunks.append(text[idx: idx + self.max_chunk_character_size])
            idx += self.max_chunk_character_size - self.overlap 
            
        chunks = [{"chunk_idx": chunk_idx, "unofficial_text": chunk, **text_metadata} for chunk_idx, chunk in enumerate(text_chunks)]
        return chunks
    
class TopicChunker(Chunker):
    
    def __init__(self, threshold: float = 0.6, max_chunk_character_size: int = 1024, sentence_emb_model=None, sentence_splitter: SentenceSplitter = None):
        super().__init__()
        self.threshold = threshold
        self.max_chunk_character_size = max_chunk_character_size
        self.sentence_emb_model = sentence_emb_model
        self.sentence_splitter = sentence_splitter
        
    def measure_similarities(self, text):
        sentences = self.sentence_splitter.split(text)

        # embed each sentence
        sentence_embeddings = self.sentence_emb_model.embed(sentences)
        sentence_embeddings = torch.Tensor(sentence_embeddings)  # [num_sentences, embedding_dim]
        
        # get cosine similarity of adjacent sentences by creating a copy and shifting it then applying cosine similarity for vectorized operation
        staggered_sentence_embeddings = torch.roll(sentence_embeddings, -1, dims=0)
        cosine_similarities = F.cosine_similarity(sentence_embeddings, staggered_sentence_embeddings, dim = 1)
        
        return cosine_similarities.tolist()
        
        

    def chunk(self, text: str, text_metadata: dict = {}) -> list[str]:

        # Assumes text is split into sentences some way
        sentences = self.sentence_splitter.split(text)
        # if len(sentences) > 128:
        #     # TODO remove, this is for OOM
        #     sentences = sentences[:128]

        # embed each sentence
        sentence_embeddings = self.sentence_emb_model.embed(sentences)
        sentence_embeddings = torch.Tensor(sentence_embeddings)  # [num_sentences, embedding_dim]
        
        # compute cosine similarity between adjacent sentences and merge sentences until similarity drops below threshold or max chunk size is reached
        text_chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        # get cosine similarity of adjacent sentences by creating a copy and shifting it then applying cosine similarity for vectorized operation
        staggered_sentence_embeddings = torch.roll(sentence_embeddings, -1, dims=0)
        cosine_similarities = F.cosine_similarity(sentence_embeddings, staggered_sentence_embeddings, dim = 1)
        # cosine_similarities = cosine_similarities[:-1] # drop similarity between first and last sentence
        # breakpoint()

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_chunk_size += len(sentence)

            # if last sentence, finalize current chunk
            if i == len(sentences) - 1:
                text_chunks.append(" ".join(current_chunk))
                break

            # if similarity drops below threshold or max_chunk_size exceeded, finalize chunk
            if cosine_similarities[i] < self.threshold or current_chunk_size >= self.max_chunk_character_size:
                text_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_size = 0

        # optionally attach metadata to chunks (if needed)
        chunks = [{"chunk_idx": chunk_idx, "unofficial_text": chunk, **text_metadata} for chunk_idx, chunk in enumerate(text_chunks)]

        return chunks
    
if __name__ == "__main__":
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # tokenizer = model.tokenizer
    text = "This is a sample document. It contains multiple sentences. The purpose is to test the TopicChunker. We want to see how well it can chunk text based on topic boundaries. Each chunk should ideally not exceed the maximum chunk size specified.\
        Let's add more sentences to ensure we have enough content to create multiple chunks. This will help us evaluate the chunking process effectively.\
        Finally, we will print out the resulting chunks to verify their sizes and contents.\
        Here is another sentence to make sure we have enough data. The quick brown fox jumps over the lazy dog. Natural language processing is a fascinating field of study.\
        Chunking text is an important step in many NLP applications."
    # Embed with topic chunker
    sentence_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    # document_embedding_model_name = "Qwen/Qwen3-Embedding-8B"
    # logger.info(f"Initializing sentence and document models: {sentence_embedding_model_name}, {document_embedding_model_name}")

    qwen_sentence_model = HFSentenceEmbeddingModelWrapper(sentence_embedding_model_name)
    # qwen_document_model = HFDocumentEmbeddingModelWrapper(document_embedding_model_name)
    # qwen_document_model = HFSentenceEmbeddingModelWrapper(document_embedding_model_name)
    logger.info("Models initialized")

    splitter = SentenceSplitter()
    chunker = TopicChunker(threshold=0.6, max_chunk_character_size=1024, sentence_emb_model=qwen_sentence_model, sentence_splitter=splitter)
    # chunker = TopicChunker(tokenizer, topic_model=sentence_embedding_model_name, max_chunk_size=50)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk: {chunk}\n")
        # print(f"Chunk {i+1}: {chunk['unofficial_text_en']}\n")
    print(f"Total Chunks: {len(chunks)}")
    breakpoint()
    
#     def testNaiveChunker():
#         chunker = NaiveChunker(max_chunk_character_size=10, overlap=2)
#         text = "Legal documents are unstructured, use legal\
# jargon, and have considerable length, making\
# them difficult to process automatically via conventional text processing techniques. A legal\
# document processing system would benefit substantially if the documents could be segmented\
# into coherent information units. This paper\
# proposes a new corpus of legal documents annotated (with the help of legal experts) with\
# a set of 13 semantically coherent units labels\
# (referred to as Rhetorical Roles), e.g., facts, arguments, statute, issue, precedent, ruling, and\
# ratio. We perform a thorough analysis of the\
# corpus and the annotations. For automatically\
# segmenting the legal documents, we experiment with the task of rhetorical role prediction:\
# given a document, predict the text segments\
# corresponding to various roles. Using the created corpus, we experiment extensively with\
# various deep learn"
#         chunks = chunker.chunk(text)
#         for chunk in chunks:
#             print(f"{chunk}")
#     testNaiveChunker()
    