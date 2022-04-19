#%%
import submitit
from datasets import load_from_disk
from datasets import concatenate_datasets
import string
import os
from random import randint

n_rows = 6078422 #len(wiki)
n_splits = 200
split_indeces_list = list(range(n_splits))


def main(split_indeces):
    wiki = load_from_disk('/storage/harai/wiki/silver')
    
    wiki_split=wiki.shard(num_shards=n_splits, index=split_indeces)

    #%%

    def add_tags(entry):
            """
            mutate a dictionary entry to have all mutations
            """

            try:
                lower = []
                punc = []
                punc_pos = []
                split = []
                split_pos = []
                redup = []
                redup_pos = []
                all = []
                all_pos = []
                for (i,word) in enumerate(entry['text']):

                    # entry is a tokenized word
                    pos_word = entry['pos'][i]
                    dice_roll = randint(1, 20)
                    word_lower = word.lower()
                    # initialize default
                    p = [word_lower]
                    a= [word_lower]
                    s =  [word]
                    r = [word]
                    pp, sp, rp, ap=  [pos_word],  [pos_word],  [pos_word],  [pos_word]
                                
                    # decapitalize
                    lower.append(word_lower)
                                
                    # depunctuate
                    if word in string.punctuation:
                        p = pp = []

                    # add random punctuation/newline/reduplication
                    if dice_roll ==20:
                        s = [word, '.' ]
                        sp = [pos_word, '.']

                        a = [word_lower, '.' ]
                        ap = [pos_word, '.']
                        

                    elif dice_roll in [1,2,3]:
                        s = [word, '\n\n']
                        sp = [pos_word, '_SP']

                        a = [word_lower, '\n\n']
                        ap =[pos_word, '_SP']

                    elif dice_roll in [5,6,7]:
                        # redup
                        r = [word, word]
                        rp = [pos_word, pos_word]

                        a = [word_lower, word_lower]
                        ap = [pos_word, pos_word]


                    for (i, item) in enumerate([p, pp, s, sp, r, rp, a, ap]):
                        [punc, punc_pos, split, split_pos, redup, redup_pos, all, all_pos][i]+=item
                        
                entry['lower'] = lower
                entry['punc'] = punc
                entry['punc_pos'] = punc_pos
                entry['split'] = split
                entry['split_pos'] = split_pos
                entry['redup'] = redup
                entry['redup_pos'] = redup_pos
                entry['all'] = all
                entry['all_pos'] = all_pos
    # pos, text, title
                return entry
            except BaseException as e:
                print("Something crashed somewhere")
                return {'error': str(e)}


    wiki_silver = wiki_split.map(add_tags)
    print("well the split is done")
    storage=os.environ.get('STORAGE')
    data_dir = "wiki/split_poetry"
    out_dataset_name = "wiki_poetry{}".format(split_indeces)
    dataset_path = os.path.join(storage, data_dir,out_dataset_name)
    wiki_silver.save_to_disk(dataset_path)

    return wiki_silver

print("submitting")
executor = submitit.AutoExecutor(folder="log_poetry_wiki")

executor.update_parameters(timeout_min=60*24)
executor.update_parameters(slurm_array_parallelism=36)

print(split_indeces_list)
print("submitting2")
jobs = executor.map_array(main, split_indeces_list[100:136]) 

print([job.job_id for job in jobs])  # ID of your job

# result_array = [job.result() for job in jobs]
# # concatenating
# print("concat")
# whole_wiki = concatenate_datasets(result_array)
# storage=os.environ.get('STORAGE')
# data_dir = "wiki/split2"
# out_dataset_name = "wiki_silver_WHOLE"
# dataset_path = os.path.join(storage, data_dir,out_dataset_name)
# whole_wiki.save_to_disk(dataset_path)


