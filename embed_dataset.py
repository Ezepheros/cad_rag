import sys
import os
from utils.logging_util import get_logger, setup_logger, log_resource_usage, start_resource_monitor, stop_resource_monitor
setup_logger(log_file="embedding_test.log", level="DEBUG")
logger = get_logger(__name__)

cad_rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

logger.info(f"Adding {cad_rag_path} to sys.path")
sys.path.append(cad_rag_path)

from data.Dataset import CanadianCaseLawDocumentDatabase
from embedding_models.embedding_models_wrapper import EmbeddingModelWrapper, HFSentenceEmbeddingModelWrapper, HFDocumentEmbeddingModelWrapper, DDPHFSentenceEmbeddingModelWrapper
from indexing.splitter import SentenceSplitter
from indexing.chunker import TopicChunker, NaiveChunker, Chunker

from tqdm import tqdm
import torch

import pyarrow as pa
import pyarrow.parquet as pq

import pandas as pd
from pathlib import Path
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModel

def save_chunks(all_chunks: list[dict], output_path: os.PathLike, dataset_name: str):
    
    chunked_df = pd.DataFrame(all_chunks)

    logger.info(f"Saving {len(all_chunks)} chunks...")
    # save first 50 entries to a csv for quick inspection
    sample_file_path = output_path / f"{dataset_name}_embedded_chunked_sample.csv"
    chunked_df.head(50).to_csv(sample_file_path, index=False)

    parquet_file_path = output_path / f"{dataset_name}_embedded_chunked.parquet"
    pickle_file_path = output_path / f"{dataset_name}_embedded_chunked.pkl"

    # Write to parquet
    writer = None

    for batch_df in pd.read_csv(..., chunksize=10000):
        table = pa.Table.from_pandas(batch_df)
        if writer is None:
            writer = pq.ParquetWriter(parquet_file_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()
    # chunked_df.to_parquet(parquet_file_path, engine='pyarrow', index=False)
    chunked_df.to_pickle(pickle_file_path)
    logger.info(f"Embedded chunked data saved to {parquet_file_path} and {pickle_file_path}")
    

def embed_dataset(dataset: CanadianCaseLawDocumentDatabase, embedding_model: EmbeddingModelWrapper, chunker: Chunker, output_dir: str, dataset_name: str, document_batch_size: int = 32, en_only=True):
    import pandas as pd
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Embedding dataset: {dataset_name}")
    all_chunks = []
    batch_texts = []
    batch_metadatas = []
    doc_idx = 0
    doc_idx_en = 0
    doc_idx_fr = 0
    debugging_cols = ['citation_en', 'name_en', 'document_date_en', 'citation_fr', 'name_fr', 'document_date_fr']
    for i, row in enumerate(tqdm(dataset)):
        text_en = row['unofficial_text_en']
        text_fr = row['unofficial_text_fr']
        if en_only and (not text_en or not text_en.strip()):
            logger.warning(f"Skipping empty text for document index {doc_idx}, metadata: { {col: row[col] for col in debugging_cols} }")
            continue

        metadata = {
            col: row[col] for col in row if col != 'unofficial_text_en' and col != 'unofficial_text_fr'
        }
        metadata['doc_idx'] = doc_idx
        metadata['doc_idx_en'] = doc_idx_en
        metadata['doc_idx_fr'] = doc_idx_fr
        doc_idx += 1
        
        if text_en is not None and len(text_en) > 0:
            metadata['chunk_language'] = 'en'
            batch_texts.append(text_en)
            batch_metadatas.append(metadata)
            doc_idx_en += 1
        
        if text_fr is not None and len(text_fr) > 0:
            metadata['chunk_language'] = 'fr'
            batch_texts.append(text_fr)
            batch_metadatas.append(metadata)
            doc_idx_fr += 1
        
        if (i + 1) % document_batch_size == 0:
            log_resource_usage(logger)
            # log amount of characters and words in batch
            total_characters = sum(len(text) for text in batch_texts)
            total_words = sum(len(text.split()) for text in batch_texts)
            logger.info(f"Processing batch of {document_batch_size} documents | Total Documents Processed: {doc_idx + 1} | Total characters: {total_characters} | Total approximate words: {total_words}")

            batch_chunks = chunker.batch_chunk(batch_texts, batch_metadatas)
            
            # batch embed each chunk
            chunk_embeddings = []
            texts_to_embed = []
            for chunks in batch_chunks:
                for chunk in chunks:
                    texts_to_embed.append(chunk['unofficial_text'])

            logger.info(f"embedding {len(texts_to_embed)} chunks...")
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

        if (i + 1) % (document_batch_size * 20) == 0:
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
    RUN_DIR = "./runs/"
    OUTPUT_DIR = RUN_DIR + "embedded_naive_chunked_ONCA_512-64_QWEN8B"
    # OUTPUT_DIR = "./TEST"

    logger.info(f"NUM GPUS IN EMBEDDER: {torch.cuda.device_count()}")
    logger.info("Getting ONCA dataset")
    data = CanadianCaseLawDocumentDatabase()
    onca_data = data.get_dataset_by_name("ONCA")
    
    # TODO make arguments and log them
    
    # Embed with topic chunker
    # sentence_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    # document_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    # logger.info(f"Initializing sentence and document models: {sentence_embedding_model_name}, {document_embedding_model_name}")

    # qwen_sentence_model = HFSentenceEmbeddingModelWrapper(sentence_embedding_model_name)
    # # qwen_document_model = HFDocumentEmbeddingModelWrapper(document_embedding_model_name)
    # qwen_document_model = HFSentenceEmbeddingModelWrapper(document_embedding_model_name)
    # logger.info("Models initialized")

    # splitter = SentenceSplitter()
    # chunker = TopicChunker(threshold=0.6, max_chunk_character_size=1024, sentence_emb_model=qwen_sentence_model, sentence_splitter=splitter)
    # logger.info("Splitter and Chunker initialized")

    # embed_dataset(data, qwen_document_model, chunker, output_dir="./embedded_chunked_data", dataset_name="ONCA", document_batch_size=48)

    # Embed with Naive Chunker
    document_embedding_model_name = "Qwen/Qwen3-Embedding-8B"
    logger.info(f"Initializing document model: {document_embedding_model_name}")
    qwen_document_model = HFSentenceEmbeddingModelWrapper(document_embedding_model_name)
    logger.info(f"{document_embedding_model_name} Model initialized")

    max_chunk_character_size = 512
    overlap = 64
    chunker = NaiveChunker(max_chunk_character_size=max_chunk_character_size, overlap=overlap)

    start_resource_monitor()

    embed_dataset(
        onca_data, 
        qwen_document_model, 
        chunker, 
        output_dir=OUTPUT_DIR, 
        dataset_name="ONCA", 
        document_batch_size=64
    )







