import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import copy
import torch.nn.functional as F

import numpy as np

from tqdm.notebook import tqdm
from collections import OrderedDict

class FCNetwork(nn.Module):
    """Fully Connected Network Class"""
    def __init__(self, n_input, layers, n_output, act=('relu', nn.ReLU())):
        """
        A simple class to create a fully connected network with the activation of 
        your choice. 
        
        :param n_input: Integer. Size of input vector.
        :param layers: Tuple. containing the desired hidden layer architecture. Each
            element i of the tuple is the number of nodes desired for ith hidden
            layer.
        :param n_output: Size of output vector.
        :param act: Tuple ('name', act_func). The first element should be a string
            describing the activation function. The second element is the activation
            function itself. Default is ``('ReLU', nn.ReLU())``.
        """
        super().__init__()
        self.input = nn.Linear(n_input, layers[0])
        self.hidden = self.init_hidden(layers, activation=act)
        self.output = nn.Linear(layers[-1], n_output)
        
    def init_hidden(self, layers, activation, dropout=0.0):

        n_layers = len(layers)
        modules = OrderedDict()
        a_name = activation[0]
            
        modules[f'{a_name}_in'] = activation[1]
        
        for i in range(n_layers - 1):
            modules[f'fc{i}'] = nn.Linear(layers[i], layers[i + 1])
            modules[f'{a_name}{i}'] = activation[1]
            modules[f'drop{i}'] = nn.Dropout(p=dropout)
            
        modules[f'{a_name}_out'] = activation[1]
        
        return nn.Sequential(modules)
            
    def forward(self, x):
        x = x.float()
        x = self.input(x)
        x = self.hidden(x)
        return self.output(x)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        

class LSTMEncoderDecoder(nn.Module):
    """
    LSTM Based Encoder Decoder. Architecture drawn from: https://arxiv.org/pdf/1607.00148.pdf
    
    Encoder-Decoder that encodes sequence data via an LSTM's final hidden state. Decoding is
    then performed using a single LSTMCell in combination with a dense layer in the 
    following way: 

    For original sequence length of ``N``:
    1. Get Decoder initial hidden state ``hs[N]``: Just use encoder final hidden state.
    2. Reconstruct last element in the sequence: ``x[N]= w.dot(hs[N]) + b``.
    2. Same pattern for other elements: ``x[i]= w.dot(hs[i]) + b``
    3. use ``x[i]`` and ``hs[i]`` as inputs to ``LSTMCell`` to get ``x[i-1]`` and ``hs[i-1]``
    """
    
    
    def __init__(self, n_features, emb_size):
        """
        :param n_features: The number of variables per time-step of the sequence.
        :param emb_size: Size of the latent space.
        
        """
        super(LSTMEncoderDecoder, self).__init__()
        self.n_features = n_features
        self.emb_size = emb_size
        self.hidden_size = emb_size

        self.encoder = SeqEncoderLSTM(n_features, emb_size)
        self.decoder = SeqDecoderLSTM(emb_size, n_features)
    
    def forward(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = x.to(device)
        
        seq_len = x.shape[1]
        hs = self.encoder(x)
        hs = tuple([h.view(-1, self.emb_size) for h in hs])
        out = self.decoder(hs, seq_len)
        return out #.unsqueeze(0):

class SeqEncoderLSTM(nn.Module):
    """
    Sequence Encoder. To be joined and trained with an accompanying Decoder. 
    """
    def __init__(self, n_features, latent_size):
        """
        :param n_features: The number of features/variable in a time step.
        :param latent_size: Size of the encoded latent space. 
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            n_features, 
            latent_size, 
            batch_first=True)
        
    def forward(self, x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = x.to(device)
        
        _, hs = self.lstm(x)
        return hs
        
        
class SeqDecoderLSTM(nn.Module):
    """
    Sequence decoder class. To be joined and trained with a sequence encoder.
    """
    def __init__(self, emb_size, n_features):
        """
        :param emb_size: The size of the latent space vectors being decoded.
        :param n_features: The number of variables/features in the original unencoded data. 
        """
        super().__init__()
        
        self.features = n_features
        self.cell = nn.LSTMCell(n_features, emb_size)
        self.dense = nn.Linear(emb_size, n_features)
        
        
    def forward(self, hs_0, seq_len):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # add each point to the sequence as it's reconstructed
        x = torch.tensor([]).to(device)
        
        # Final hidden and cell state from encoder
        hs_i, cs_i = hs_0
        
        # reconstruct first (last) element
        x_i = self.dense(hs_i)
        x = torch.cat([x, x_i])
        
        # reconstruct remaining elements
        for i in range(1, seq_len):
            hs_i, cs_i = self.cell(x_i, (hs_i, cs_i))
            x_i = self.dense(hs_i)
            x = torch.cat([x, x_i])
        return x.view(-1, seq_len, self.features)

    
def train_encoder(model, epochs, trainload, testload=None, criterion=nn.MSELoss(), optimizer=optim.Adam, 
                  lr=1e-6, es_patience=20, reverse=False, LSTM=True):
    """
    Train auto-encoder reconstruction for given number of epochs.
    
    :param model: Encoder or autoencoder model to train.
    
    :param epochs: Number of times the network will view the entire data set.
    
    :param trainload: A DataLoader object containing training variables and targets 
        used for training.
        
    :param testload: Optional. a DataLoader containing the validation set. 
    
    :param criterion: Objective function.
    
    :param optimizer: Learning method. Default optim.Adam.
    
    :param lr: Learning Rate. Default 1e-6.
    
    :param es_patience: Early stop patience. If validation loss does not improve after
        this many epochs, halt training and load previous best parameters. 
        
    :param LSTM: If the autoencoder is LSTM based an additional dimension needs to be added
        to ensure model input has shape (batch_size, seq_len, n_features), which may 
        not occur if the time series is univariate. 
        
    :return loss: Tuple of lists containing training history ``(train_hist, valid_hist)``. 
        If no validation data is provided, ``valid_hist`` will be an empty list.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training model on {device}')
    model = model.to(device)
    opt = optimizer(model.parameters(), lr)

    train_loss = []
    valid_loss = []
    best_loss = np.inf
    curr_patience = es_patience
    
    for e in tqdm(range(epochs)):
        
        tl, vl = train_one_epoch(model, opt, criterion, trainload, testload, LSTM)
        train_loss.append(tl / len(trainload))
        
        if testload is not None:
            valid_loss.append(vl / len(testload))
            
            if valid_loss[-1] < best_loss:
                best_loss = valid_loss[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                curr_patience = es_patience
            else:
                curr_patience -= 1

        if curr_patience == 0:
            print(f'No improvement in {es_patience} epochs. Interrupting training.')
            print(f'Best loss: {best_loss}')
            print(f'Loading best model weights.')
            model.load_state_dict(best_model_wts)
            print('Training complete.')
            break
        
    return train_loss, valid_loss

def train_one_epoch(model, opt, criterion, trainload, testload, LSTM):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    running_tl = 0
    running_vl = 0
    
    for x in trainload:
        
        # inference
        if LSTM: 
            x = x.view(x.shape[0], x.shape[1], -1) 
        x = x.to(device).float()
        opt.zero_grad()
        x_hat = model(x)
        
        # back-prop
        loss = criterion(x_hat, x)
        loss.backward()
        opt.step()
        running_tl += loss.item()

    if testload is not None:
        model.eval()
        with torch.no_grad():
            for x in testload:
                if LSTM:
                    x = x.view(x.shape[0], x.shape[1], -1) 
                x = x.to(device).float()
                loss = criterion(model(x), x)
                running_vl += loss.item()    
        model.train() 
        
    return running_tl, running_vl