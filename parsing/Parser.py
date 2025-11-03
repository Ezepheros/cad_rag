from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import re
from enum import Enum
import os
import sys

class PARAGRAPH_STYLE(Enum):
    BRACKETED = 0
    SPACE = 1

class Parser:
    """
        Parses parquet files to extract meta data and court case paragraphs
    """
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.pq_file = pq.ParquetFile(self.file_path)
        
        self.error_log_path = Path(os.path.dirname(self.file_path)) / "parsing_errors.log"
        os.makedirs(os.path.dirname(self.error_log_path), exist_ok=True)
        self.error_log_dict = {"citation_en": [],
                             "citation_fr": [],
                             "url_en": [], 
                             "error_message": []}
        
    def log_error(self, row: pd.Series, message: str):
        self.error_log_dict["citation_en"].append(row['citation_en'])
        self.error_log_dict["citation_fr"].append(row['citation_fr'])
        self.error_log_dict["url_en"].append(row['url_en'])
        self.error_log_dict["error_message"].append(message)
        
    def parse(self, batch_size: int = 4096):
        columns = self.pq_file.schema.names
        for batch in self.pq_file.iter_batches(batch_size=batch_size):
            breakpoint()
            batch_df = batch.to_pandas()
            for _, row in batch_df.iterrows():
                # read 'unofficial_text_en' if null, set to empty string
                text = row['unofficial_text_en'] if pd.notnull(row['unofficial_text_en']) else ""
                if text == "":
                    continue
                
                # Get subject meta data from text
                # TODO
                
                # find first occurrence of paragraph markers like '[1], [2]' or '1 ', '2 ' or ' [1] ', ' [2] ', etc.
                pattern = r'(\s\[\d+\][ ])|(\s\d+[ ])'
                matches = list(re.finditer(pattern, text))
                if not matches:
                    self.log_error(row, "No paragraph markers found")
                    continue
                
                paragraph_style = PARAGRAPH_STYLE.BRACKETED if matches[0].group(1) else PARAGRAPH_STYLE.SPACE
                
                # Split text into paragraphs based on the identified style
                breakpoint()
    
        # save error log
        if self.error_log_dict["file_path"]:
            error_log_df = pd.DataFrame(self.error_log_dict)
            error_log_df.to_csv(self.error_log_path, index=False, mode='a', header=True)
        
                
            
        
if __name__ == "__main__":
    parser = Parser("../data/canadian-case-law/CHRT/train.parquet")
    parser.parse()