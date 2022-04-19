
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy import data

import spacy
import numpy as np

import time
import random
from tqdm import tqdm
import pandas as pd

# %%
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# # %%
# from datasets import load_dataset, load_from_disk
import os
import random
# from spacy.util import minibatch, compounding
from pathlib import Path

def main():


    storage=os.environ.get('STORAGE')
    data_dir = "wiki"
    out_dataset_name = "silver"
    model_name = "fromzero"
    json_file = "silversomehowthisworks.json"
    dataset_path = os.path.join(storage, data_dir,out_dataset_name)
    model_out_path = os.path.join(storage, data_dir,model_name)
    silver_json_path = os.path.join(storage, data_dir,json_file)


    # train = os.path.join(storage, data_dir,"silver_trainn.json")
    # test = os.path.join(storage, data_dir,"silver_test.json")
    # vali = os.path.join(storage, data_dir,"silver_vali.json")


    # %%
    # create Field objects
    # TITLE = data.Field()
    TEXT = data.Field(lower=False)
    POS = data.Field(unk_token = None)

    # create a dictionary representing the dataset
    fields = {
    'text': ('text', TEXT),
    'pos': ('pos', POS)
    }

    # load the dataset in json format
    train_ds, test_ds, vali_ds = data.TabularDataset.splits(
    path = '/storage/harai/wiki/try',
    train = 'small_train.json',
    test = 'small_test.json',
    validation = 'small_vali.json',
    format = 'json',
    fields = fields
    )

  
    MIN_FREQ = 2

    TEXT.build_vocab(train_ds, 
                    min_freq = MIN_FREQ,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)


    POS.build_vocab(train_ds)



    # %%
    def tag_percentage(tag_counts):
        
        total_count = sum([count for tag, count in tag_counts])
        
        tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]
            
        return tag_counts_percentages



    for tag, count, percent in tag_percentage(POS.vocab.freqs.most_common()):
        print(f"{tag}\t\t{count}\t\t{percent*100:4.1f}%")

    BATCH_SIZE = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # %%
    class BiLSTMPOSTagger(nn.Module):
        def __init__(self, 
                    input_dim, 
                    embedding_dim, 
                    hidden_dim, 
                    output_dim, 
                    n_layers, 
                    bidirectional, 
                    dropout, 
                    pad_idx):
            
            super().__init__()
            
            self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
            
            self.lstm = nn.LSTM(embedding_dim, 
                                hidden_dim, 
                                num_layers = n_layers, 
                                bidirectional = bidirectional,
                                dropout = dropout if n_layers > 1 else 0)
            
            self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
            
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, text):

            #text = [sent len, batch size]
            
            #pass text through embedding layer
            embedded = self.dropout(self.embedding(text))
            
            #embedded = [sent len, batch size, emb dim]
            
            #pass embeddings into LSTM
            outputs, (hidden, cell) = self.lstm(embedded)
            
            #outputs holds the backward and forward hidden states in the final layer
            #hidden and cell are the backward and forward hidden and cell states at the final time-step
            
            #output = [sent len, batch size, hid dim * n directions]
            #hidden/cell = [n layers * n directions, batch size, hid dim]
            
            #we use our outputs to make a prediction of what the tag should be
            predictions = self.fc(self.dropout(outputs))
            
            #predictions = [sent len, batch size, output dim]
            
            return predictions

   
# %%
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(POS.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = BiLSTMPOSTagger(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)

    # %% [markdown]
    # We initialize the weights from a simple Normal distribution. Again, there may be a better initialization scheme for this model and dataset.

    # %%
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.normal_(param.data, mean = 0, std = 0.1)
            
    model.apply(init_weights)

    # %% [markdown]
    # Next, a small function to tell us how many parameters are in our model. Useful for comparing different models.

    # %%
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # %% [markdown]
    # We'll now initialize our model's embedding layer with the pre-trained embedding values we loaded earlier.
    # 
    # This is done by getting them from the vocab's `.vectors` attribute and then performing a `.copy` to overwrite the embedding layer's current weights.

    # %%
    pretrained_embeddings = TEXT.vocab.vectors

    print(pretrained_embeddings.shape)

    # %%
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # %% [markdown]
    # It's common to initialize the embedding of the pad token to all zeros. This, along with setting the `padding_idx` in the model's embedding layer, means that the embedding should always output a tensor full of zeros when a pad token is input.

    # %%
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print(model.embedding.weight.data)

    # %% [markdown]
    # We then define our optimizer, used to update our parameters w.r.t. their gradients. We use Adam with the default learning rate.

    # %%
    optimizer = optim.Adam(model.parameters())

    # %% [markdown]
    # Next, we define our loss function, cross-entropy loss.
    # 
    # Even though we have no `<unk>` tokens within our tag vocab, we still have `<pad>` tokens. This is because all sentences within a batch need to be the same size. However, we don't want to calculate the loss when the target is a `<pad>` token as we aren't training our model to recognize padding tokens.
    # 
    # We handle this by setting the `ignore_index` in our loss function to the index of the padding token in our tag vocabulary.

    # %%
    TAG_PAD_IDX = POS.vocab.stoi[POS.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    # %% [markdown]
    # We then place our model and loss function on our GPU, if we have one.

    # %%
    model = model.to(device)
    criterion = criterion.to(device)

    # %%
    # print(device)

    # %% [markdown]
    # We will be using the loss value between our predicted and actual tags to train the network, but ideally we'd like a more interpretable way to see how well our model is doing - accuracy.
    # 
    # The issue is that we don't want to calculate accuracy over the `<pad>` tokens as we aren't interested in predicting them.
    # 
    # The function below only calculates accuracy over non-padded tokens. `non_pad_elements` is a tensor containing the indices of the non-pad tokens within an input batch. We then compare the predictions of those elements with the labels to get a count of how many predictions were correct. We then divide this by the number of non-pad elements to get our accuracy value over the batch.

    # %%
    def categorical_accuracy(preds, y, tag_pad_idx):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
        non_pad_elements = (y != tag_pad_idx).nonzero()
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        return correct.sum() / y[non_pad_elements].shape[0]


    # %%
    def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()
        
        for batch in tqdm(iterator):
            
            text = batch.text
            tags = batch.pos
            
            optimizer.zero_grad()
            
            #text = [sent len, batch size]
            
            predictions = model(text)
            
            #predictions = [sent len, batch size, output dim]
            #tags = [sent len, batch size]
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            #predictions = [sent len * batch size, output dim]
            #tags = [sent len * batch size]
            
            loss = criterion(predictions, tags)
                    
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # %%
    def evaluate(model, iterator, criterion, tag_pad_idx):
        
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
        
            for batch in tqdm(iterator):

                text = batch.text
                tags = batch.pos
                
                predictions = model(text)
                
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                
                loss = criterion(predictions, tags)
                
                acc = categorical_accuracy(predictions, tags, tag_pad_idx)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_diff(model, iterator, criterion, tag_pad_idx):
    
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
        
            for batch in tqdm(iterator):

                text = batch.all
                tags = batch.all_pos
                
                predictions = model(text)
                
                predictions = predictions.view(-1, predictions.shape[-1])
                tags = tags.view(-1)
                
                loss = criterion(predictions, tags)
                
                acc = categorical_accuracy(predictions, tags, tag_pad_idx)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    # %%
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


        
    # IMPORTANT PARTS
    print("we got here")
    ALL = data.Field(lower=False)
    ALL_POS = data.Field(unk_token = None)
    # NEWTEXT = data.Field(lower=False)
    # NEW_POS = data.Field(unk_token = None)

    poetryfields = {
    # 'text': ('text', NEWTEXT),
    # 'pos': ('pos', NEW_POS),
    'redup': ('all', ALL),
    'redup_pos': ('all_pos', ALL_POS)
    }

                #     entry['lower'] = lower
                # entry['punc'] = punc
                # entry['punc_pos'] = punc_pos
                # entry['split'] = split
                # entry['split_pos'] = split_pos
                # entry['redup'] = redup
                # entry['redup_pos'] = redup_pos
                # entry['all'] = all
                # entry['all_pos'] = all_pos

    poetry = data.TabularDataset(
    path = '/storage/harai/wiki/smaller_silverpoetry.json',
    format = 'json',
    fields = poetryfields
    )
    MIN_FREQ = 2


    
    poetry_train_ds, poetry_test_ds, poetry_vali_ds = poetry.split([8,1,1])

    smaller_test_ds, not_used = test_ds.split(split_ratio = 0.02)
    del not_used
    ALL.build_vocab(poetry_train_ds, 
                    min_freq = MIN_FREQ,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)


    ALL_POS.build_vocab(poetry_train_ds)

    print(len(poetry_train_ds), len(poetry_test_ds), len(poetry_vali_ds))
    print("length of poetry and original " +str(len(poetry_test_ds)) + " " + str(len(smaller_test_ds)))


    # %%

    orig_test_iterator = data.BucketIterator(
        smaller_test_ds, 
        # sort_key=lambda x: len(x.comment_text),
        sort = False,
        batch_size = BATCH_SIZE,
        device = device)

    poetry_test_iterator = data.BucketIterator(
        poetry_test_ds,
        sort= False,
        batch_size = BATCH_SIZE,
        device = device
    )

    # train_size = len(smaller_train_ds)

    # N_EPOCHS = 10
    # score_list = []

    # best_valid_loss = float('inf')

    # for epoch in range(N_EPOCHS):
    #     torch.cuda.empty_cache()
    #     start_time = time.time()
        
    #     train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
    #     valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
        
    #     end_time = time.time()

    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), '/storage/harai/wiki/models/ow_size{}-model.pt'.format(size*100))
        
    #     print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    #     score_list.append({'train_size': train_size, 
    #                     'size':size,
    #                     'epoch': epoch,
    #                     'epoch_min': epoch_mins,
    #                     'epoch_sec': epoch_secs,
    #                     'train_loss': train_loss,
    #                     'train_acc': train_acc,
    #                     'vali_loss': valid_loss,
    #                     'vali_acc': valid_acc,
    #                     })
                        


    # %% [markdown]
    # We then load our "best" parameters and evaluate performance on the test set.
    model.load_state_dict(torch.load('/storage/harai/wiki/models/ow_epoch9-model.pt'))

    test_loss, test_acc = evaluate_diff(model, poetry_test_iterator, criterion, TAG_PAD_IDX)

    print(f'Poetry redup Test Loss: {test_loss:.3f} |  Poetry redup Test Acc: {test_acc*100:.2f}%')


    test_loss, test_acc = evaluate(model, orig_test_iterator, criterion, TAG_PAD_IDX)

    print(f'Original Test Loss: {test_loss:.3f} |  Original Test Acc: {test_acc*100:.2f}%')



    # %% [markdown]
    # ## Inference
    # 
    # 88% accuracy looks pretty good, but let's see our model tag some actual sentences.
    # 
    # We define a `tag_sentence` function that will:
    # - put the model into evaluation mode
    # - tokenize the sentence with spaCy if it is not a list
    # - lowercase the tokens if the `Field` did
    # - numericalize the tokens using the vocabulary
    # - find out which tokens are not in the vocabulary, i.e. are `<unk>` tokens
    # - convert the numericalized tokens into a tensor and add a batch dimension
    # - feed the tensor into the model
    # - get the predictions over the sentence
    # - convert the predictions into readable tags
    # 
    # As well as returning the tokens and tags, it also returns which tokens were `<unk>` tokens.

    # %%
    def tag_sentence(model, device, sentence, text_field, tag_field):
        
        model.eval()
        
        if isinstance(sentence, str):
            nlp = spacy.load('en_core_web_sm')
            tokens = [token.text for token in nlp(sentence)]
        else:
            tokens = [token for token in sentence]

        if text_field.lower:
            tokens = [t.lower() for t in tokens]
            
        numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]

        unk_idx = text_field.vocab.stoi[text_field.unk_token]
        
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        
        token_tensor = torch.LongTensor(numericalized_tokens)
        
        token_tensor = token_tensor.unsqueeze(-1).to(device)
            
        predictions = model(token_tensor)
        
        top_predictions = predictions.argmax(-1)
        
        predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
        
        return tokens, predicted_tags, unks

    # %% [markdown]
    # We'll get an already tokenized example from the training set and test our model's performance.



    example_index = 1

    sentence = vars(poetry_test_ds.examples[example_index])['all']
    actual_tags = vars(poetry_test_ds.examples[example_index])['all_pos']

    print(sentence)

    # %% [markdown]
    # We can then use our `tag_sentence` function to get the tags. Notice how the tokens referring to subject of the sentence, the "respected cleric", are both `<unk>` tokens!

    # %%
    tokens, pred_tags, unks = tag_sentence(model, 
                                        device, 
                                        sentence, 
                                        TEXT, 
                                        POS)

    print(unks)

    # %% [markdown]
    # We can then check how well it did. Surprisingly, it got every token correct, including the two that were unknown tokens!

    # %%
    print("Pred. Tag\tActual Tag\tCorrect?\tToken\n")

    for token, pred_tag, actual_tag in zip(tokens, pred_tags, actual_tags):
        correct = '✔' if pred_tag == actual_tag else '✘'
        print(f"{pred_tag}\t\t{actual_tag}\t\t{correct}\t\t{token}")




        
    # %%
    example_index = 1

    sentence = vars(train_ds.examples[example_index])['text']
    actual_tags = vars(train_ds.examples[example_index])['pos']

    print(sentence)

    # %% [markdown]
    # We can then use our `tag_sentence` function to get the tags. Notice how the tokens referring to subject of the sentence, the "respected cleric", are both `<unk>` tokens!

    # %%
    tokens, pred_tags, unks = tag_sentence(model, 
                                        device, 
                                        sentence, 
                                        TEXT, 
                                        POS)

    print(unks)

    # %% [markdown]
    # We can then check how well it did. Surprisingly, it got every token correct, including the two that were unknown tokens!

    # %%
    print("Pred. Tag\tActual Tag\tCorrect?\tToken\n")

    for token, pred_tag, actual_tag in zip(tokens, pred_tags, actual_tags):
        correct = '✔' if pred_tag == actual_tag else '✘'
        print(f"{pred_tag}\t\t{actual_tag}\t\t{correct}\t\t{token}")

    # # %% [markdown]
    # # Let's now make up our own sentence and see how well the model does.
    # # 
    # # Our example sentence below has every token within the model's vocabulary.

    # # %%
    # sentence = 'The Queen will deliver a speech about the conflict in North Korea at 1pm tomorrow.'

    # tokens, tags, unks = tag_sentence(model, 
    #                                 device, 
    #                                 sentence, 
    #                                 TEXT, 
    #                                 POS)

    # print(unks)

    # %% [markdown]
    # Looking at the sentence it seems like it gave sensible tags to every token!

    # %%
    # print("Pred. Tag\tToken\n")

    # for token, tag in zip(tokens, tags):
    #     print(f"{tag}\t\t{token}")





import submitit


print("submitting")
executorbig = submitit.AutoExecutor(folder="test_poetry")

executorbig.update_parameters(timeout_min=60*24, 
                            slurm_partition="gpu-standard",
                            # gres="gpu:1",
                            )
print("submitting2")

jobbig = executorbig.submit(main) 
print(jobbig.job_id)  # ID of your job