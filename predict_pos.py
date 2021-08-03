#%%
import pandas as pd
import numpy as np
import typing as T
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import json
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder

@dataclass
class Word:
    original: str
    pos: str
    normalized: str
    glove: T.List[float]

RANDOM_SEED = 1234
GLOVE_LENGTH = 50
#%%import glove vectors

glove_dict = {}
with open("../data/glove.6B.50d.txt", "rt",encoding="utf8") as vecf:
    for line in tqdm(vecf, total=400000):
        split = line.index(" ")
        word = line[:split]
        vector = np.fromstring(line[split + 1 :], dtype=np.float32, sep=" ")
        glove_dict[word] = vector

def get_embedding(word: str, pos) -> T.List[float]:
    if word in glove_dict:
        return glove_dict[word]
    else:
        return np.zeros(GLOVE_LENGTH)


#%%# import sentences

file_path = "data/wiki_labels.json"
num_lines = sum(1 for line in open(file_path,'r', errors="ignore"))
articles_list = []
with open(file_path, "rt", encoding='utf-8') as fp:
    for entry in tqdm(fp, total=num_lines):
        word_list = []
        data = json.loads(entry)
        text = data["text"]
        pos = data["pos"]
        for i, word in enumerate(text):
            word_normalized = word.lower()
            word_pos = pos[i]
            glove = get_embedding(word_normalized, word_pos)

            word_list.append(Word(word, word_pos, word_normalized, glove))
        articles_list.append(word_list)



#%% create df (keep adding to df until full) (params: n_words), each entry is a slice of the sentence of length n (eg.5) and should predict the
# pos of the middle word
# need np size of 250 in each row
GLOVE_LENGTH = 50
def slice_sentences(articles_list: T.List, slice_length: int = 3, delete_punct = True) -> T.Tuple[pd.DataFrame, np.ndarray]:
    entry_list = []
    all_glove = []
    center_index = slice_length//2 +1
    for article in articles_list:
        if delete_punct:
            article = [word for word in article if word.normalized not in punctuation]
        for i in range(len(article)//slice_length):
            start_index = slice_length*i
            slice = article[start_index:start_index+slice_length]
            vec = []
            for word in slice:
                vec += word.glove.tolist()

            entry = {
                "text": " ".join([word.normalized for word in slice]),
                "pos": slice[center_index].pos[0]
            }
            all_glove.append(np.array(vec))

            entry_list.append(entry)
    return (pd.DataFrame(entry_list), np.vstack(all_glove))

(df, W) = slice_sentences(articles_list, 5)
labels = LabelEncoder()
labels.fit(df["pos"])
df["label"] = labels.transform(df["pos"])
#%%
print(df["glove"].head(1))
df = df.astype('object')
print(len(df["glove"].head(1)))
# df.head()


#%%
list(labels.classes_)
#%% split df, prep data
def prepare_data(df: pd.DataFrame, fit: bool = False) -> T.Tuple[np.ndarray, np.ndarray]:
    global labels
    # extract truth_value as y:
    y = df.pop("label")
    X = W[df.index]
    print(X.shape)
    return y, X
    
# split the whole dataframe:
# tv_f, test_f = train_test_split(df, test_size=0.25, random_state=RANDOM_SEED)
tv_f, test_f = train_test_split(df, test_size=0.25, random_state=RANDOM_SEED)
train_f, vali_f = train_test_split(tv_f, test_size=0.25, random_state=RANDOM_SEED)

# use the 'prepare_data'
train_y, train_X = prepare_data(train_f,  fit=True)
vali_y, vali_X = prepare_data(vali_f)
test_y, test_X = prepare_data(test_f)
print(train_X.shape, train_y.shape)

#%% train & vali

mlp = MLPRegressor(random_state=RANDOM_SEED)
print(train_X.shape, train_y.shape)
mlp.fit(train_X, train_y)
# pred = mlp.predict(vali_X)

print("length 5")
print("\tTrain-AUC, 3-gram: {:.3}".format(mlp.score(train_X, train_y)))
print("\tVali-AUC, 3-gram: {:.3}".format(mlp.score(vali_X, vali_y)))


# %%
