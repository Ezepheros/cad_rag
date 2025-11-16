import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cad_rag')))
from cad_rag.data.Dataset import CanadianCaseLawDocumentDatabase
from cad_rag.embedding_models.embedding_models_wrapper import EmbeddingModelWrapper, VoyageaiEmbeddingModelWrapper, DummyEmbeddingModelWrapper
import re
from nltk.tokenize import sent_tokenize
import nltk.data
import nltk
from nltk.tokenize import word_tokenize
from cad_rag.indexing.chunker import NaiveChunker
from cad_rag.data.Dataset import Dataset
import pyarrow as pa
from pyarrow import parquet as pq
from pyarrow import dataset as ds
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()

class Indexer:
    def __init__(self):
        pass
    
class NaiveIndexer(Indexer):
    """
        Naively indexes documents by chunking them into fixed-size chunks.
    """
    
    def __init__(self, emb_model: EmbeddingModelWrapper, dataset: Dataset, chunker: NaiveChunker, pq_write_path: str, faiss_write_path: str, embedding_dim: int = 1024, included_datasets: list[str] = set(["ONCA"]), max_token_usage: int = 100_000, embeddings_per_request: int = 800):
        super().__init__()
        self.emb_model = emb_model
        self.dataset = dataset
        self.chunker = chunker
        self.pq_write_path = pq_write_path
        self.faiss_write_path = faiss_write_path
        self.emb_dim = embedding_dim
        self.included_datasets = included_datasets
        self.max_token_usage = max_token_usage
        self.embeddings_per_request = embeddings_per_request
        self.token_usage = 0
        self.index_ = None
        
        # DEBUGGING
        self.breakpoints = {}
            
    def create_index(self):
        #TODO replace unofficial_text_en with unofficial_text
        pq_writer = None
        idx = 0
        chunks = []
        
        for i, doc in enumerate(self.dataset):
                
            dataset_name = doc.get('dataset')
            if dataset_name not in self.included_datasets:
                # TODO make filter more efficient
                continue
            # if not self.breakpoints.get("exit_forloop", False):
            #     # self.breakpoints["exit_forloop"] = True
            #     breakpoint()
            doc_metadata = {doc_key: doc[doc_key] for doc_key in doc if doc_key != 'unofficial_text_en' and doc_key != 'unofficial_text_fr'}
            chunks += self.chunker.chunk(doc['unofficial_text_en'], text_metadata=doc_metadata) if doc['unofficial_text_en'] else []
            
            if len(chunks) >= self.embeddings_per_request:
                print(f"Indexing document {i}+ from dataset '{dataset_name}' with {len(chunks)} chunks...")
                
                texts = [chunk['unofficial_text_en'] for chunk in chunks]
                embeddings = self.emb_model.embed(texts)
                
                # if not self.breakpoints.get("exit_embedding", False):
                #     # self.breakpoints["exit_forloop"] = True
                #     # self.breakpoints["exit_embedding"] = True
                #     breakpoint()
                    
                for j in range(len(chunks)):
                    # if not self.breakpoints.get("exit_chunkloop", False):
                    #     # self.breakpoints["exit_forloop"] = True
                    #     # self.breakpoints["exit_embedding"] = True
                    #     # self.breakpoints["exit_chunkloop"] = True
                    #     breakpoint()
                        
                    num_tokens = len(self.chunker.tokenizer.encode(chunks[j]['unofficial_text_en']).ids)
                    self.token_usage += num_tokens
                    
                    embedding = embeddings[j].tolist()
                    emb_dim = len(embedding)
                    if self.emb_dim is None:
                        self.emb_dim = emb_dim
                    if emb_dim != self.emb_dim:
                        raise ValueError(f"Embedding dimension mismatch: expected {self.emb_dim}, got {emb_dim}")
                        
                    chunk_record = {
                        'index_id': idx,
                        'document_id': f'{dataset_name}_doc-{i}_chunk-{j}',
                        'unofficial_text_en': chunks[j]['unofficial_text_en'],
                        'embedding': embedding,
                        **{k: v for k, v in chunks[j].items() if k != 'unofficial_text_en'}
                    }
                    
                    table = pa.Table.from_pydict({k: [v] for k, v in chunk_record.items()})
                    if pq_writer is None:
                        pq_writer = pq.ParquetWriter(self.pq_write_path, table.schema)
                
                    pq_writer.write_table(table)
                    
                    if self.token_usage > self.max_token_usage:
                        print(f"Reached max token usage of {self.max_token_usage} at dataset '{dataset_name}', document {i}, chunk {j}. Stopping indexing.")
                        if pq_writer is not None:
                            pq_writer.close()
                        return
                    
                    idx += 1
                
                chunks = []
        
        # TODO fix so that you write remaining chunks after loop (refactor into function)
        
        if pq_writer is not None:
            pq_writer.close()
    
    
    def build_faiss_index(self):
        if not self.breakpoints.get("before_faiss", False):
            # self.breakpoints["before_faiss"] = True
            breakpoint()
            
        pq_file = pq.ParquetFile(self.pq_write_path)
        self.index = faiss.IndexFlatL2(self.emb_dim) # TODO change later to IVF or HNSW for large datasets, also maybe cosine similarity
        
        for batch in pq_file.iter_batches(batch_size=4096):
            embeddings = batch.column("embedding")
            embeddings_np = np.array([row.as_py() for row in embeddings])
            # try:
            #     batch_df = batch.to_pandas() # can just read embedding column instead of all columns, fix if needed TODO
            #     embeddings = np.array(batch_df['embedding'].tolist()).astype('float32')
            # except Exception as e:
            #     print(f"Error converting embeddings to numpy array: {e}")
            #     breakpoint()
            # embeddings = np.array(batch_df['embedding'].tolist()).astype('float32')
            self.index.add(embeddings_np)
        
        faiss.write_index(self.index, self.faiss_write_path)
        
    def build_index(self):
        self.create_index()
        self.build_faiss_index()
        
    def read_index(self):
        self.index_ = faiss.read_index(self.faiss_write_path)
    
    def get_knn(self, query: str, k: int = 5):
        if not self.breakpoints.get("before_knn", False):
            # self.breakpoints["before_knn"] = True
            breakpoint()
            
        query_emb = self.emb_model.embed(query)
        if self.index_ is None:
            self.read_index()
            
        D, I = self.index_.search(np.array(query_emb), k=k)
        
        # Retrieve corresponding documents from Parquet file
        dataset = ds.dataset(self.pq_write_path, format="parquet")
        
        # filter rows by index_id in I
        index_id_set = set(I[0].tolist())
        filtered = dataset.to_table(filter=ds.field("index_id").isin(index_id_set))
        return filtered.to_pandas() # TODO dont use pandas
        
if __name__ == "__main__":
    def testIndexer():
        import os
        exp_name = "naive_rag_test"
        index_path = f"../indexes/{exp_name}/"
        pq_write_path = os.path.join(index_path, "indexed_data.parquet")
        faiss_write_path = os.path.join(index_path, "faiss_index.index")
        os.makedirs(os.path.dirname(pq_write_path), exist_ok=True)
        # os.makedirs(faiss_write_path, exist_ok=True)
        
        emb_model = DummyEmbeddingModelWrapper(model_name="dummy-model", emb_dim=1024)
        
        data = CanadianCaseLawDocumentDatabase()
        chunker = NaiveChunker(max_chunk_character_size=512, overlap=64, tokenizer=emb_model.tokenizer())
        indexer = NaiveIndexer(emb_model=emb_model,
                               dataset=data,
                               chunker=chunker,
                               pq_write_path=pq_write_path,
                               faiss_write_path=faiss_write_path,
                               embedding_dim=1024)
        indexer.build_index()
        
        # breakpoint()
        results = indexer.get_knn("If I rented a property but didn't pay rent due to COVID restrictions and claim that the contract was void due to abnormal events, can the landlord kick me out?", k=5)
        print(results)
        breakpoint()
    
    testIndexer()
    
    def testVoyageIndexer():
        import voyageai as vo
        import os
        exp_name = "naive_rag_test"
        index_path = f"../indexes/{exp_name}/"
        pq_write_path = os.path.join(index_path, "indexed_data.parquet")
        faiss_write_path = os.path.join(index_path, "faiss_index.index")
        os.makedirs(pq_write_path, exist_ok=True)
        os.makedirs(faiss_write_path, exist_ok=True)
        
        vo_client = vo.Client()
        emb_model = VoyageaiEmbeddingModelWrapper(model_name="voyage-3.5", vo_client=vo_client)
        
        data = CanadianCaseLawDocumentDatabase()
        chunker = NaiveChunker(max_chunk_character_size=512, overlap=64, tokenizer=vo_client.tokenizer(model="voyage-3.5"))
        indexer = NaiveIndexer(emb_model=emb_model,
                               dataset=data,
                               chunker=chunker,
                               pq_write_path=pq_write_path,
                               faiss_write_path=faiss_write_path,
                               embedding_dim=1024)
        indexer.build_index()
        
        # breakpoint()
        results = indexer.get_knn("If I rented a property but didn't pay rent due to COVID restrictions and claim that the contract was void due to abnormal events, can the landlord kick me out?", k=5)
        print(results)
        breakpoint()
        
    # testVoyageIndexer()

