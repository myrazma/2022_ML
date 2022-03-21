# -------------------------------------
# 
# BI-LSTM for Named Entity Recognition
#
# Author: Myra Zmarsly
# 
# ------------- sources ---------------
# 
# For the basic BILSTM structure and training in PyTorch: BiLSTM - Pytorch and Keras, Rahul Agarwal, 2019,
#       url: https://www.kaggle.com/mlwhiz/bilstm-pytorch-and-keras
# For the basic BILSTM structure and training in PyTorch: Pytorch Bidirectional LSTM example, Aladdin Persson, 2020, 
#       video_url: https://www.youtube.com/watch?v=jGst43P-TJA
#       url to code on github: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_bidirectional_lstm.py
# BiLSTM for PoS Tagging, Ben Trevett, 2021,
#       url: https://github.com/bentrevett/pytorch-pos-tagging/blob/master/1_bilstm.ipynb
# For usage of pre-trained embeddings layer: Load pre-trained GloVe embeddings in torch.nn.Embedding layer… in under 2 minutes!, Tanmay Garg, 2021,
#       url: https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
# 
# -------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as func
import numpy as np
import pandas as pd
import matplotlib as plt
import time
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('device:', device)


class BILSTM(nn.Module):
    """ The class for the BI-LSTM with one layer

    Args:
        nn (nn.Module): Module from pytorch
    """

    def __init__(self, input_dim, hidden_units, output_dim, max_seq_len, embeddings):
        """Initializer for BILSTM class

        Args:
            input_dim (int): The input dimension
            hidden_units (int): [description]
            output_dim (int): [description]
            max_seq_len (int): [description]
            embeddings (np.array) [description]: 
        """
        super(BILSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.embed_vocab_size = embeddings.shape[0]  # vocabulary size of the embeddings
        self.embed_dim = embeddings.shape[1]  # vector size of the embeddings
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.layer_num = 1

        # init embeddings layer
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        self.embedding.weight.requires_grad = False  # do not update the embeddings while training
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))  # initilaize embeddings with the glove embeddings
        # init bi lstm layer
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_units, self.layer_num, bidirectional=True, batch_first=True)
        # init output layer
        self.output_layer = nn.Linear(self.hidden_units*2, output_dim)  # times 2 since bi-lstm
    

    def forward(self, input):
        """Forward pass of the model

        Args:
            input (tensor): the token ids as input for the model

        Returns:
            tensor: The prediction labels 
        """
        # use embedding layer
        embedded_input = self.embedding(input)
        # init for hidden states in lstm
        h0 = torch.randn(self.layer_num*2, input.size(0), self.hidden_units).to(device)
        c0 = torch.randn(self.layer_num*2, input.size(0), self.hidden_units).to(device)
        output, (hn, cn) = self.lstm(embedded_input, (h0, c0))
        out = self.output_layer(output)
        return out


class SeqDataset(Dataset):
    """The data set to hold the Sequence data

    Args:
        Dataset (Dataset): The torch.utils.data Dataset
    """
    def __init__(self, seq, labels):
        """Initializer for SeqDataset

        Args:
            seq (np.array) [x, y]: The sequences of shape (sample_size, max_seq_size)
            labels (np.array) [x, y]: The labels of shape (sample_size, max_seq_size)
        """
        self.labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
        self.seq = torch.from_numpy(seq).type(torch.LongTensor).to(device)

    def __len__(self):
        """Implement len function of type Dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.labels)
            
    def __getitem__(self, idx):
        """Implement get_item function of type Dataset

        Args:
            idx (int): The index of the item to get

        Returns:
            tensor [y], tensor [y]: The sequence, The labels
        """
        label = self.labels[idx]
        sequence = self.seq[idx]
        return sequence, label


def read_data(filepath):
    """Read data from .conll file

    Args:
        filepath (str): The path and filename of the file

    Returns:
        list(list(str)), list(list(str)): 
            The sequence data with a list of sequences (list of words (input tokens))
            The label data with a list of labels (list of words (labels))
    """
    with open(filepath) as f:
        seq_mat, label_mat = [], []  # create data and label matrix
        sequence, labels = [], []  # create empty lists for sequences and labels
        for line in f.readlines():
            line_ls = line.strip().split('\t')

            if len(line_ls) == 1 and line_ls[0] == '':  # end of sequence
                if len(sequence) >= 1:
                    seq_mat.append(sequence)
                    label_mat.append(labels)
                sequence, labels = [], []  # if sequence is over, reset the list to empty

            if len(line_ls) == 4:  # then it is word with features and label
                sequence.append(line_ls[0].lower())  # word of sequence in first
                labels.append(line_ls[-1])
    
    return seq_mat, label_mat


def add_padding(data, labels, max_seq_length):
    """Add padding to the sequences or crop sequences based on maximum padding length

    Args:
        data (list(list(str))): The sequences
        labels (list(list(str))): The labels
        max_seq_length (int): The maximum sequence length

    Returns:
        (list(list(str))), (list(list(str))): The padded sequence data and labels
    """
    padded_data = []
    padded_labels = []
    for idx, seq in enumerate(data):
        lab = labels[idx]  # get the corresponding labels
        pad = max_seq_length - len(seq)
        if pad >= 0:  # add padding if too short
            padded_seq = seq + ['<pad>' for i in range(pad)]
            padded_lab = lab + ['<pad-label>' for i in range(pad)]
        else: # cut sequence off, if too long
            padded_seq = seq[:max_seq_length]
            padded_lab = lab[:max_seq_length]
        # add padded sequence to data
        padded_data.append(padded_seq)
        padded_labels.append(padded_lab)

    return padded_data, padded_labels


def read_embeddings(filepath):
    """Load the embeddings and save as matrix
    create a look up table (dict) for the vocbulary

    Args:
        filepath (str): The path and filename of the embeddings file

    Returns:
        np.array [x, d], dict(str: int):
            The embeddings matrix with size [vocabulary_size, embeddings_dimension]
            The vocabulary dictionary, a lookup table for tranlasting words into ids
    """
    embedding = []
    vocab_dict = {}
    
    with open(filepath,'rt') as f:

        print('Loading embeddings...')
        for idx, line in enumerate(f.readlines()):

            line_ls = line.strip().split(' ')
            vocab_dict[line_ls[0]] = idx  # append the word
            embedding.append(line_ls[1:])  # append the embeddings
        
        embedding = np.array(embedding).astype(float)  # cast empeddings to numpy array of floats
        embed_dim = embedding.shape[1]  # get the embeddings dimension
    
        # append zero vector for non existing words 
        # (handeled differently than padding, therefore another lookup entry)
        vocab_dict['<unk>'] = embedding.shape[0]
        # use mean embedding vector of all word embeddings for unkown words
        embedding = np.vstack((embedding, np.mean(embedding, axis=0)))
        
        # at the end of the embeddings I will include zero vector for the paddings 'zeropad'
        vocab_dict['<pad>'] = embedding.shape[0]
        # use zero vector for padding
        embedding = np.vstack((embedding, np.zeros(embed_dim)))
        
        print('Loading embeddings complete!')
    return embedding, vocab_dict


def data_to_ids(data, vocab_dict):
    """Convert tokens/words from data to ids based on vocabulary dict

    Args:
        data (list(list(str))): The sequence data with words
        vocab_dict (dict(str: int)): The dict for translating the words into ids (ids based on embeddings)

    Returns:
        list(list(int)): The data translated into ids
    """
    def word_id(word):
        """Get the word id of a word

        Args:
            word (str): The word / token

        Returns:
            int: The id of the word based on the vocab dictionary
        """
        try:
            word_id = vocab_dict[word]
        except:  # if word not in dictionary use unkown <unk>
            word_id = vocab_dict['<unk>']
        return word_id
    # map the function on the data
    data_id = list(map(lambda row: list(map(word_id, row)), data))
    return data_id


def get_words_from_id(ids, vocab_dict):
    """Get words from data ids

    Args:
        ids (list(list(int))): The data as ids of words
        vocab_dict (dict(str: int)): The dict for translating the words into ids (ids based on embeddings)

    Returns:
        (list(list(str))): The sequence data as words
    """
    values = list(vocab_dict.values())
    keys = list(vocab_dict.keys())
    data_translated = list(map(lambda row: list(map(lambda x: keys[values.index(int(x))], row)), ids))
    return data_translated


def create_labels_lookup_table(labels):
    """Create a lokup table for the labels, the entries will look like f.e. {'O': 9}

    Args:
        labels (list(list(str))): The data of the 'raw' / textual labels

    Returns:
        dict(str: int):
            The labels dictionary, a lookup table for tranlasting labels into ids
    """
    # make a distinct set of all labels available, make shure they are sorted, so the label idx wont change everytime
    labels_flatten = [word_lab for seq_lab in labels for word_lab in seq_lab]  
    labels_distinct = sorted(set(labels_flatten))

    # create dictionary look up table from those labels
    label_lookup = {}
    for idx, lab in enumerate(labels_distinct):
        label_lookup[lab] = idx  # word (str): id (int)
    return label_lookup


def labels_to_ids(labels, label_lookup):
    """Convert tokens/words of the labels to ids based on label_lookup dict

    Args:
        labels (list(list(str))): The sequence data with words
        label_lookup (dict(str: int)): The dict for translating the labels into ids

    Returns:
        list(list(int)): The labels translated into ids
    """
    def label_id(lab):
        try:
            lab_id = label_lookup[lab]
        except:
            lab_id = label_lookup['<unk>']
        return lab_id
    labels_id = list(map(lambda row: list(map(label_id, row)), labels))
    return labels_id


def get_labels_from_one_hot(labels_oh, label_lookup):
    """Get the labels from one hot encoding

    Args:
        labels_oh (np.array) [x, y, z]: One hot labels
        label_lookup (dict(str: int)): The dict for translating the labels into ids

    Returns:
        np.array [x, y]: The labels from one hot encoding
    """
    labels_ids = labels_oh.argmax(axis=-1)
    values = list(label_lookup.values())
    keys = list(label_lookup.keys())
    labels_translated = list(map(lambda row: list(map(lambda x: keys[values.index(int(x))], row)), labels_ids))
    return labels_translated


def one_hot_labels(labels, output_dim):
    """Get the one hot labels from the 1-dim labels

    Args:
        labels (np.array) [x, y]: The labels
        output_dim (int): The output dimension

    Returns:
        np.array [x, y, output_dim]: The labels as one hot encoding
    """
    one_hot = (np.arange(output_dim) == labels[...,None]).astype(float)
    return one_hot 


def train(model, train_loader, dev_loader, epochs, optimizer, criterion, pad_label_id, output_dim):
    """Train the model on train dataset and evelautate on the dev dataset

    Args:
        model (BILSTM): The model that should be trained
        train_loader (DataLoader): The DataLoader of the train SeqDataset
        dev_loader (DataLoader): The DataLoader of the dev SeqDataset
        epochs (int): The number of epochs
        optimizer (Optimizer): The optimizer object
        criterion (nn.CrossEntropyLoss): The Loss function, here cross-entropy loss
        pad_label_id (int): The label id of the padding label (based on label lookup table)
        output_dim (int): The output dimension of the model

    Returns:
        BILSTM: The trained model
    """
    print("Start training...\n")

    macro_f1_dev = []
    for epoch_i in range(epochs):
        # -------------------
        #      Training 
        # -------------------
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_batch = time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_count = 0, 0, 0

        
        model.train()

        # For each batch of training data...
        for batch_idx, (data, labels) in enumerate(train_loader):
            batch_count +=1
            scores = model(data)
            
            # make predictions into right shape
            predictions = scores.view(-1, scores.shape[-1])
            tags = labels.view(-1)
    
            loss = criterion(predictions, tags)
            batch_loss += loss.item()
            total_loss += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()
        
            # Print the loss values and time elapsed for every 1000 batches
            if (batch_idx % 1000 == 0 and batch_idx != 0) or (batch_idx == len(train_loader) - 1):
                # Calculate time elapsed
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {batch_idx:^7} | {batch_loss / batch_count:^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_count = 0, 0
                t0_batch = time.time()
            
        # -------------------
        #     Validation 
        # -------------------
        # calculate and print the f1 score
        macro_f1_whole_epoch = evaluate(model, dev_loader, output_dim, pad_label_id)
        macro_f1_dev.append(macro_f1_whole_epoch)
        print('Epoch {}, Macro F1 score on dev: '.format(epoch_i + 1) + f'{macro_f1_whole_epoch:.5f}')
    
    return model, macro_f1_dev


def macro_f1(y, y_hat, pad_dim):
    """Calculate macro f1 score for given one-hot labels ignoring the padding label

    Args:
        y (np.array) [x, 1, max_seq_len, output_dim]: The actual label
        y_hat (np.array) [x, 1, max_seq_len, output_dim]: The predicted label
                

    Returns:
        float: The macro f1 score
    """
    output_dim = y_hat.shape[-1]

    # exclude padding from calculating f1 score
    idx_no_pad = np.where(y.argmax(axis=-1)!=pad_dim)
    y_no_pad = y[idx_no_pad]
    y_hat_no_pad = y_hat[idx_no_pad]

    # get tp, fp and fn
    tp_occurrence = np.where((y_no_pad == 1) & (y_hat_no_pad == 1))[-1]  # last array indicating the class, where correct label was found
    tp = np.bincount(tp_occurrence, minlength=output_dim).astype(float)  # count occurences of each label
    fp_occurrence = np.where((y_no_pad == 0) & (y_hat_no_pad == 1))[-1]  # last array indicating the class, where correct label was found
    fp = np.bincount(fp_occurrence, minlength=output_dim).astype(float)  # count occurences of each label
    fn_occurrence = np.where((y_no_pad == 1) & (y_hat_no_pad == 0))[-1]  # last array indicating the class, where correct label was found
    fn = np.bincount(fn_occurrence, minlength=output_dim).astype(float)  # count occurences of each label
    
    # claculate f1 for each class
    numerator = tp
    denominator = (tp + (1/2)*(fp + fn))

    f1_label = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    # delete pad label and ignore for f1
    f1_label = np.delete(f1_label, pad_dim)
    
    # calculate average macro f1
    f1_macro = np.sum(f1_label) / f1_label.shape[0]
    return f1_macro


def evaluate(model, data_loader, output_dim, pad_label_id):
    """Evaluate the model based on the data in data loader
    Calculates the macro f1 score

    Args:
        model (BILSTM): The trained BILSTM model
        data_loader (DataLoader): The data loader including data from dataset type SeqDataset
        output_dim (int): The output dimension for the labels
        pad_label_id (int): The id of the passing label

    Returns:
        float: The macro f1 score of this evaluation
    """
    model.eval()  # set model into evaluation mode
    all_predictions, all_labels = [], []
    # Evaluate test data
    for data, labels in data_loader:
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            y_pred_proba = model(data)
        
        # tensors to numpy array for calculaitng f1 score
        y_pred_proba_np = y_pred_proba.detach().cpu().numpy()  
        labels_np = labels.detach().cpu().numpy()
        labels_one_hot = one_hot_labels(labels_np, output_dim)

        y_pred = np.zeros(y_pred_proba_np.shape)
        y_pred[np.where(y_pred_proba_np >= np.amax(y_pred_proba_np, axis=-1, keepdims=True))] = 1
        # append each prediction and label to overall dataset
        all_predictions.append(y_pred)
        all_labels.append(labels_one_hot)

    # calculate f1 score based on all predictions and labels
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    macro_f1_complete_test = macro_f1(all_labels, all_predictions, pad_label_id)

    return macro_f1_complete_test


def plot_f1_score(macro_f1, base_dir):
    """Plot the f1 score

    Args:
        macro_f1 (np.array): The macro f1 score per epoch
        base_dir (str): The path of the base directory
    """
    epochs = np.arange(len(macro_f1)).astype(int) + 1
    plt.plot(epochs , macro_f1)
    plt.title('Macro F1 score per epoch')
    plt.grid(True)
    plt.xticks(epochs)
    plt.ylabel('macro F1 score')
    plt.xlabel('epochs')
    plt.savefig(base_dir + 'plot_macro_f1_score.pdf')
    plt.show()


def run(base_dir='', do_savings=False, show_plot=False):
    """Run the complete process of the model which includes
        1. Preparing / loading the data and embeddings
        2. Training the model
        3. Evaluating the model 

    Args:
        base_dir (str, optional): The path to the base directory, if you are working from 
                                'python_test_master' this can be left to default. Defaults to ''.
        do_savings (bool, optional): If True, the macro F1 score on the dev data during
                                training will be saves as .npy file and the model will be stores. 
                                Defaults to False.
        show_plot (bool, optional): If True, the plot of the macro f1 score will be created and saved in base_dir. Defaults to False.

    Returns:
        BILSTM: The trained BILSTM model
    """

    # ---- define parameters ----
    batch_size = 1
    learning_rate = 0.001
    hidden_units = 100
    epochs = 20

    # -------------------
    #    Prepare Data 
    # -------------------

    # ---- load data ----
    train_data, train_labels = read_data(base_dir + "data/train.conll")
    dev_data, dev_labels = read_data(base_dir + "data/dev.conll")
    test_data, test_labels = read_data(base_dir + "data/test.conll")

    # ---- add padding ----
    # use padding for sequences since RNN with fixed sequence size
    # print(max([len(train_data[i]) for i in range(len(train_data))])) 
    # max length is 113, therefore I will use padding for 128
    max_seq_length = 128
    train_data, train_labels = add_padding(train_data, train_labels, max_seq_length)
    dev_data, dev_labels = add_padding(dev_data, dev_labels, max_seq_length)
    test_data, test_labels = add_padding(test_data, test_labels, max_seq_length)

    # ---- Load embeddings ----
    # downloaded here: https://nextcloud.ukp.informatik.tu-darmstadt.de/index.php/s/g6xfciqqQbw99Xw
    embeddings, vocab_dict = read_embeddings(base_dir + 'glove.6B.50d.txt')

    # ---- transform tokens/words to ids ----
    # get ids of tokens/words based on look up table of the embeddings
    train_data = data_to_ids(train_data, vocab_dict)
    dev_data = data_to_ids(dev_data, vocab_dict)
    test_data = data_to_ids(test_data, vocab_dict)
    
    # ---- Transform train_labels into numbers and create lookup table for labels ----
    # create lookup table for labels
    label_lookup = create_labels_lookup_table(train_labels)
    pad_label_id = int(label_lookup['<pad-label>'])
    
    # get output dimension (based on available labels)
    output_dim = len(label_lookup.keys())

    # translate labels into ids
    train_labels = labels_to_ids(train_labels, label_lookup)
    dev_labels = labels_to_ids(dev_labels, label_lookup)
    test_labels = labels_to_ids(test_labels, label_lookup)

    # ---- convert data into numpy array ----
    train_labels = np.array(train_labels).astype(int)
    train_data = np.array(train_data).astype(int)
    dev_labels = np.array(dev_labels).astype(int)
    dev_data = np.array(dev_data).astype(int)
    test_labels = np.array(test_labels).astype(int)
    test_data = np.array(test_data).astype(int)

    # ---- create Dataset and Data Loader ----
    train_dataset = SeqDataset(train_data, train_labels)  # create dataset
    dev_dataset = SeqDataset(dev_data, dev_labels)  # create dataset
    test_dataset = SeqDataset(test_data, test_labels)  # create dataset

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # create dataloader
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)  # create dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)  # create dataloader

    # -------------------
    #    Train Model 
    # -------------------
    # keep track of overall training time
    training_time_t0 = time.time()

    # ---- create model ----
    model = BILSTM(input_dim=max_seq_length, hidden_units=hidden_units, output_dim=output_dim, max_seq_len=max_seq_length, embeddings=embeddings)
    model.to(device)

    # init loss and ignore padding label (don't use padding for updating the model)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_label_id)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---- train model and evaluate on dev ----
    model, avg_macro_f1 = train(model, train_loader, dev_loader, epochs, optimizer, criterion, pad_label_id, output_dim)
    if do_savings: 
        np.save(base_dir + 'macro_f1_dev.npy', avg_macro_f1)
        torch.save(model.state_dict(), base_dir + 'NER_model.pth')


    # -------------------
    #   Evaluate Model 
    # -------------------

    # ---- macro-averaged F1 score on test data with final model ----
    macro_f1_test = evaluate(model, test_loader, output_dim, pad_label_id)
    print('Test Data, Macro F1 score: ' + f'{macro_f1_test:.5f}')

    # print training time needed for whole training and evalutation
    training_time_t1 = time.time()
    training_time = training_time_t1 - training_time_t0
    print('\n Training time needed:', training_time)

    # ---- plot f1 score in each epoch ----
    if show_plot: plot_f1_score(avg_macro_f1, base_dir)

    return model


if __name__ == "__main__":
    model_NER = run()