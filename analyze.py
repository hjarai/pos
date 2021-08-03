import gzip, json
from tqdm import tqdm
import re
import spacy
nlp = spacy.load("en_core_web_sm")


file_path = "data/wiki2020.00{}.tok.gz".format(str(49))
num_lines = sum(1 for line in open(file_path,'r', errors="ignore"))
WORDS = re.compile("(\w+)") 

with open("data/wiki_labels.json", "w") as out_file:
    with gzip.open(file_path, "rt", encoding='utf-8') as fp:
        for entry in tqdm(fp, total=num_lines):
            # array of strings
            data = json.loads(entry)
            sentences = data["sentences"]
            spacy_text, spacy_pos = [], []
            for sentence in sentences:
                tokens = nlp(" ".join(WORDS.findall(sentence)))
                for token in tokens:
                    spacy_text.append(token.text)
                    spacy_pos.append(token.tag_)

            new_entry = {"text": spacy_text,
                        "pos": spacy_pos}
            out_file.write(json.dumps(new_entry)+"\n") #add
