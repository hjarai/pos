#%%
import submitit
from datasets import load_dataset
from datasets import concatenate_datasets
import spacy
import os


n_rows = 6078422 #len(wiki)
n_splits = 36
split_indeces_list = list(range(n_splits))


def main(split_indeces):
    wiki = load_dataset("wikipedia", "20200501.en", split='train')
    wiki_split=wiki.shard(num_shards=n_splits, index=split_indeces)
    
    nlp = spacy.load("en_core_web_sm")
    #%%
    def add_tags(entry):
        """
        mutate a dictionary entry to have a row of pos tags
        """
        try:
            spacy_analyzed = nlp(entry['text'])
            entry['pos'] = [token.tag_ for token in spacy_analyzed]
            entry['text'] = [token.text for token in spacy_analyzed]
            return entry
        except BaseException as e:
            print("Something crashed somewhere")
            return {'error': str(e)}


    wiki_silver = wiki_split.map(add_tags)
    print("well the split is done")
    storage=os.environ.get('STORAGE')
    data_dir = "wiki/split2"
    out_dataset_name = "wiki_silver{}".format(split_indeces)
    dataset_path = os.path.join(storage, data_dir,out_dataset_name)
    wiki_silver.save_to_disk(dataset_path)

    return wiki_silver

print("submitting")
executor = submitit.AutoExecutor(folder="log_pos_wiki")

executor.update_parameters(timeout_min=60*24)
executor.update_parameters(slurm_array_parallelism=36)

print(split_indeces_list)
print("submitting2")
jobs = executor.map_array(main, split_indeces_list[6:]) 

print([job.job_id for job in jobs])  # ID of your job

result_array = [job.result() for job in jobs]
# concatenating
print("concat")
whole_wiki = concatenate_datasets(result_array)
storage=os.environ.get('STORAGE')
data_dir = "wiki/split2"
out_dataset_name = "wiki_silver_WHOLE"
dataset_path = os.path.join(storage, data_dir,out_dataset_name)
whole_wiki.save_to_disk(dataset_path)


