# this creates the validation set for retrieval by getting chunks from the retreival set and creating queries from chunks which can be answered by those chunks
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cad_rag.data.Dataset import CanadianCaseLawDocumentDatabase
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / '..' / 'runs' / 'embedded_naive_chunked_ONCA_512-64_QWEN8B' / 'ONCA_embedded_chunked.pkl'
    PROMPT_PATH = Path(__file__).parent / 'prompts' / 'create_questions.txt'
    NUM_SAMPLES = 100
    NUM_QUERIES_PER_CHUNK = 5

    df = pd.read_pickle(DATA_PATH)
    print(f"loaded {len(df)} chunks")

    sampled_chunks = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"sampled {len(sampled_chunks)} chunks")

    template = PROMPT_PATH.read_text()
    queries = []

    for idx, row in sampled_chunks.iterrows():
        doc = row['unofficial_text']
        prompt = template.format(
            document_text=doc,
            num_queries=NUM_QUERIES_PER_CHUNK
        )

        # call LLM to generate queries
        

    

# 1. Get full retreival set

# 2. sample chunks

# 3. create queries from each sampled chunk

# 4. save set