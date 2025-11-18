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

def process_texts(onca_data, chunker, save_iters=8192, save_path="./runs/similarities/"):
    similarities = []
    
    for i, entry in enumerate(onca_data):
        text = entry['unofficial_text_en']
        if not text:
            continue
            
        text_sims = chunker.measure_similarities(text)
        similarities += text_sims # text_sims is a list
        
        if (i + 1) % save_iters == 0:
            os.mkdir(save_path, exist_ok=True)
            df = pd.DataFrame({'similarities': similarities})
            
            logger.info(f"saving length {len(df)} df to path '{save_path}'...")
            df.to_pickle(save_path)
            
    os.mkdir(save_path, exist_ok=True)
    df = pd.DataFrame({'similarities': similarities})
    
    logger.info(f"saving df to path '{save_path}'...")
    df.to_pickle(save_path)

if __name__ == "__main__":
    RUN_DIR = "./runs/"
    DATASET_NAME = "ONCA"
    OUTPUT_DIR = RUN_DIR + f"Sentence_similarities_{DATASET_NAME}_QWEN-0.6B/"
    save_path = OUTPUT_DIR + "similarities.pickle"
    # OUTPUT_DIR = "./TEST"

    logger.info(f"NUM GPUS IN EMBEDDER: {torch.cuda.device_count()}")
    logger.info("Getting ONCA dataset")
    data = CanadianCaseLawDocumentDatabase()
    onca_data = data.get_dataset_by_name(DATASET_NAME)
    
    # Embed with topic chunker
    sentence_embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    qwen_sentence_model = HFSentenceEmbeddingModelWrapper(sentence_embedding_model_name)
    logger.info("Models initialized")

    splitter = SentenceSplitter()
    chunker = TopicChunker(threshold=0.6, max_chunk_character_size=1024, sentence_emb_model=qwen_sentence_model, sentence_splitter=splitter)
    
    process_texts(onca_data, chunker, save_path=OUTPUT_DIR)