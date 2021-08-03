from tqdm import tqdm
from dataclasses import dataclass
import json


@dataclass
class Word:
    original: str
    pos: str
    normalized: str
    

file_path = "data/wiki_labels.json"
num_lines = sum(1 for line in open(file_path,'r', errors="ignore"))

with open(file_path, "rt", encoding='utf-8') as fp:
    for entry in tqdm(fp, total=num_lines):
        data = json.loads(entry)
        text = data["text"]
        pos = data["pos"]
        # uhh
        sentence_array = []
        for i, word in enumerate(text):
            sentence_array.append(Word(word, pos[i], word.lower()))


"""  lowercasing (or random capitalization?)
getting rid of punctuation
throwing in misspellings
slicing sentences randomly
"""
