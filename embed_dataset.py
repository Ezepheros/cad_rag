import sys
import os
from utils.logging_util import get_logger, setup_logger
setup_logger(log_file="embedding_test.log", level="DEBUG")
logger = get_logger(__name__)

cad_rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
logger.info(f"Adding {cad_rag_path} to sys.path")
sys.path.append(cad_rag_path)

from data.Dataset import CanadianCaseLawDocumentDatabase
from embedding_models.embedding_models_wrapper import EmbeddingModelWrapper, HFSentenceEmbeddingModelWrapper, HFDocumentEmbeddingModelWrapper
from indexing.splitter import SentenceSplitter
from indexing.chunker import TopicChunker, NaiveChunker, Chunker

from tqdm import tqdm

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel



def save_chunks(all_chunks: list[dict], output_path: os.PathLike, dataset_name: str):
    import pandas as pd
    from pathlib import Path
    chunked_df = pd.DataFrame(all_chunks)

    logger.info(f"Saving {len(all_chunks)} chunks...")
    # save first 50 entries to a csv for quick inspection
    sample_file_path = output_path / f"{dataset_name}_embedded_chunked_sample.csv"
    chunked_df.head(50).to_csv(sample_file_path, index=False)

    chunked_file_path = output_path / f"{dataset_name}_embedded_chunked.parquet"
    pickle_file_path = output_path / f"{dataset_name}_embedded_chunked.pkl"

    chunked_df.to_parquet(chunked_file_path, engine='pyarrow', index=False)
    chunked_df.to_pickle(pickle_file_path)
    logger.info(f"Embedded chunked data saved to {chunked_file_path} and {pickle_file_path}")
    

def embed_dataset(dataset: CanadianCaseLawDocumentDatabase, embedding_model: EmbeddingModelWrapper, chunker: Chunker, output_dir: str, dataset_name: str, document_batch_size: int = 32):
    import pandas as pd
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Embedding dataset: {dataset_name}")
    all_chunks = []
    batch_texts = []
    batch_metadatas = []
    doc_idx = 0
    debugging_cols = ['citation_en', 'name_en', 'document_date_en', 'citation_fr', 'name_fr', 'document_date_fr']
    for i, row in enumerate(tqdm(dataset)):
        text = row['unofficial_text_en']
        if not text or not text.strip():
            logger.warning(f"Skipping empty text for document index {doc_idx}, metadata: { {col: row[col] for col in debugging_cols} }")
            continue

        metadata = {
            col: row[col] for col in row if col != 'unofficial_text_en' and col != 'unofficial_text_fr'
        }
        metadata['doc_idx'] = doc_idx
        doc_idx += 1

        batch_texts.append(text)
        batch_metadatas.append(metadata)
        
        if (i + 1) % document_batch_size == 0:
            # log amount of characters and words in batch
            total_characters = sum(len(text) for text in batch_texts)
            total_words = sum(len(text.split()) for text in batch_texts)
            logger.info(f"Processing batch of {document_batch_size} documents | Total characters: {total_characters} | Total approximate words: {total_words}")

            batch_chunks = chunker.batch_chunk(batch_texts, batch_metadatas)
            
            # batch embed each chunk
            chunk_embeddings = []
            texts_to_embed = []
            for chunks in batch_chunks:
                for chunk in chunks:
                    texts_to_embed.append(chunk['unofficial_text_en'])

            embeddings = embedding_model.embed(texts_to_embed)
            emb_idx = 0
            for chunks in batch_chunks:
                for chunk in chunks:
                    chunk['embedding'] = embeddings[emb_idx]
                    emb_idx += 1

            for chunks in batch_chunks:
                all_chunks.extend(chunks)

            batch_texts = []
            batch_metadatas = []
            logger.info(f"Processed {i + 1} documents")

        if (i + 1) % (document_batch_size * 10) == 0:
            # save intermediate results
            logger.info(f"Saving intermediate results after processing {i + 1} documents")
            save_chunks(all_chunks, output_path, dataset_name)
    
    # process remaining documents
    if batch_texts:
        batch_chunks = chunker.batch_chunk(batch_texts, batch_metadatas)
        for chunks in batch_chunks:
            all_chunks.extend(chunks)
    
    save_chunks(all_chunks, output_path, dataset_name)

if __name__ == "__main__":
    logger.info("Getting ONCA dataset")
    data = CanadianCaseLawDocumentDatabase()
    onca_data = data.get_dataset_by_name("ONCA")
    
    sentence_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    document_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    logger.info(f"Initializing sentence and document models: {sentence_embedding_model_name}, {document_embedding_model_name}")

    qwen_sentence_model = HFSentenceEmbeddingModelWrapper(sentence_embedding_model_name)
    # qwen_document_model = HFDocumentEmbeddingModelWrapper(document_embedding_model_name)
    qwen_document_model = HFSentenceEmbeddingModelWrapper(document_embedding_model_name)
    logger.info("Models initialized")

    splitter = SentenceSplitter()
    chunker = TopicChunker(threshold=0.6, max_chunk_character_size=1024, sentence_emb_model=qwen_sentence_model, sentence_splitter=splitter)
    logger.info("Splitter and Chunker initialized")

    embed_dataset(data, qwen_document_model, chunker, output_dir="./embedded_chunked_data", dataset_name="ONCA", document_batch_size=48)






