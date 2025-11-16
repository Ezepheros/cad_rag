import os
import pandas as pd
from pyarrow import parquet as pq

class Dataset:
    def __init__(self):
        pass

class CanadianCaseLawDocumentDatabase(Dataset):
    def __init__(self):
        super().__init__()
        
        self.data_dir = os.path.join('..', os.path.dirname(__file__), 'canadian-case-law') # ./path/to/canadian-case-law
        self.dataset_names = self.get_dataset_names()
        self.dataset_dirs = [os.path.join(self.data_dir, name) for name in self.dataset_names] # ex. ['./canadian-case-law/ONCA', './canadian-case-law/CHRT', ...]
        
    def get_dataset_names(self):
        names = []
        for dir_ in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, dir_)) and not dir_.startswith('.'):
                names.append(dir_)
        return names
    
    def get_dataset_as_df(self):
        # Returns a generator of dataframes
        for dataset_dir in self.dataset_dirs:
            yield pd.read_parquet(os.path.join(dataset_dir, 'train.parquet'), engine='pyarrow')
            
    def get_dataset_by_name_as_df(self, name):
        if name not in self.dataset_names:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {self.dataset_names}")
        return pd.read_parquet(os.path.join(self.data_dir, name, 'train.parquet'), engine='pyarrow')
    
    def get_dataset_by_name(self, name):
        return self._ds_iter(name)

    def _ds_iter(self, name):
        pq_file = pq.ParquetFile(os.path.join(self.data_dir, name,'train.parquet'))
        for batch in pq_file.iter_batches(batch_size=4096):
            batch_df = batch.to_pydict()
            for i in range(len(batch_df['citation_en'])):
                row = {key: batch_df[key][i] for key in batch_df}
                yield row
    
    def __iter__(self):
        for dataset_path in self.dataset_dirs:
            pq_file = pq.ParquetFile(os.path.join(dataset_path, 'train.parquet'))
            for batch in pq_file.iter_batches(batch_size=4096):
                batch_df = batch.to_pydict()
                for i in range(len(batch_df['citation_en'])):
                    row = {key: batch_df[key][i] for key in batch_df}
                    yield row
                   
if __name__ == "__main__":
    def testCanadianCaseLawDocumentDatabase():
        data = CanadianCaseLawDocumentDatabase()
        print("'data' is a dataframe")
        df1 = data.get_dataset_by_name_as_df('ONCA')
        print(df1)
        for ds in data.get_dataset():
            print(ds)
            breakpoint()
            
    def testIterator():
        data = CanadianCaseLawDocumentDatabase()
        for i, row in enumerate(data):
            breakpoint()
            print(row)
            if i >= 10:
                break
    
    def testIteratorWhole():
        data = CanadianCaseLawDocumentDatabase()
        count = 0
        for i, row in enumerate(data):
            if (i % 5000) == 0:
                print(f'Row {i}: {row["citation_en"]}, name: {row["name_en"]}')
            count += 1
        print(f'Total rows: {count}')
        breakpoint()
    
    # testIteratorWhole()
    # testIterator()
    # testCanadianCaseLawDocumentDatabase()