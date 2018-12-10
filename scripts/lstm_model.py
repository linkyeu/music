from common import *

#-----------------------------------------------------------------
class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.6):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        return hidden
    
#------------------------------------------------------------------------

def row2embeding_vector(row):
    return net.embedding(torch.Tensor(row).long()).detach().numpy().reshape(-1)

def row2average_embeding_vector(row):
    x = net.embedding(torch.Tensor(row).long()).detach().numpy()
    return x.mean(0)

#----------------------------------------------------------------------------

def load_embeddings(average=True, model='model_lstm'):
    net = torch.load(model)
    net.cpu()
    
    X = np.nan_to_num(train_df[sites].values)
    X_test = np.nan_to_num(test_df[sites].values)
    
    func = row2average_embeding_vector if average else row2embeding_vector        

    X_embeds = []
    for o in tqdm(X):
        X_embeds.append(func(o))

    X_test_embeds = []
    for o in tqdm(X_test):
        X_test_embeds.append(func(o))

    X_embeds = np.array(X_embeds)
    X_test_embeds = np.array(X_test_embeds)
    return X_embeds, X_test_embeds