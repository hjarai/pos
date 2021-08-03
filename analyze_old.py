import gzip, json
from nltk.metrics.aline import diff
from tqdm import tqdm
import numpy as np
from collections import Counter

import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

file_path = "data/wiki2020.00{}.tok.gz".format(str(49))
differences = []
with gzip.open(file_path, "rt", encoding='utf-8') as fp:
    for entry in tqdm(fp):
        # array of strings
        data = json.loads(entry)
        sentences = data["sentences"]

        for sentence in sentences:

            # spacy analysis
            spacy_analyzed = nlp(sentence)
            spacy_pos = [token.tag_ for token in spacy_analyzed]
            spacy_text = [token.text for token in spacy_analyzed]

            # nltk analysis
            nltk_analyzed = nltk.pos_tag(spacy_text)
            nltk_pos = [token[1] for token in nltk_analyzed]
            nltk_text = [token[0] for token in nltk_analyzed]

            # core-nlp analysis???

        
            # # master tag
            # pos = []
            for (i, tags) in enumerate(zip(spacy_pos, nltk_pos)):
                if tags[0] != tags[1]:
                    differences.append(" ".join([tags[0], tags[1]]))
Counter(differences).most_common()

# using spacy


# out_dict = {'text': text, 'pos' : pos}
# with open('data/wiki_labelled', 'w') as out_file:
#     json.dump(out_dict, out_file)
# glove = {}
# with open("../data/glove.6B.50d.txt", "rt",encoding="utf8") as vecf:
#     for line in tqdm(vecf, total=400000):
#         split = line.index(" ")
#         word = line[:split]
#         vector = np.fromstring(line[split + 1 :], dtype=np.float32, sep=" ")
#         glove[word] = vector


# doc = nlp("He was a genius of the best kind and his dog was green.")
# for chunk in doc.noun_chunks:  
#     spacy.displacy.serve(doc, style='dep')