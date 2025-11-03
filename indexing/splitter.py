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

class Splitter:
    def __init__(self):
        pass
    
class WhiteSpaceSplitter(Splitter):
    """
        Splits documents into paragraphs based on whitespace
    """
    
    def __init__(self):
        super().__init__()
        
    def split(self, text: str) -> list[str]:
        pattern = r"\s"
        paragraphs = []
        start_idx = 0
        end_idx = 0
        in_middle_word = False
        
        for char in text:
            if not re.match(pattern, char):
                if not in_middle_word:
                    end_idx = start_idx
                    
                end_idx += 1
                in_middle_word = True
                
            else:
                if in_middle_word:
                    paragraphs.append(text[start_idx:end_idx])
                    start_idx = end_idx + 1
                else:
                    start_idx += 1
                in_middle_word = False
                
            # print(f"char: '{char}' | start_idx: {start_idx} | end_idx: {end_idx} | in_middle_word: {in_middle_word}")
            breakpoint()
        
        return paragraphs
    
            
    
    # def split(self, text: str) -> list[str]:
    #     pattern = r"\s\S+\s"
    #     matches = re.finditer(pattern, text)
    #     paragraphs = []
    #     for match in matches:
    #         paragraphs.append(match.group())
    #     return paragraphs

if __name__ == "__main__":
    path = Path('./cad_rag/text_samples/CHRT_sample_0.txt')
    sample_text = path.read_text(encoding='utf-8')
    
    splitter = WhiteSpaceSplitter()
    chunks = splitter.split(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{chunk}\n")