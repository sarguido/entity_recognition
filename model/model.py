import torch
import torch.nn as nn

from torch.autograd import Variable
from typing import List, Dict, Union


class BidirectionalLSTM(nn.Module):
    """
    Class for the bi-directional LSTM model.
    """

    def __init__(self,
                 vocab_size: int,
                 tag_size: int,
                 embedding_dim: int,
                 n_hidden: int,
                 batch_size: int,
                 dropout_p: float):
        """
        Initialize the model with the parameter set.

        :param vocab_size: Number of words in the vocabulary.
        :param tag_size: Number of labels in the vocabulary.
        :param embedding_dim: Dimension for embedding layer.
        :param n_hidden: Number of features in the hidden state.
        :param batch_size: Size of the input batch.
        :param dropout_p: Dropout probability for regularization.
        """
        super(BidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=n_hidden,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.n_labels = tag_size
        self.hidden2label = nn.Linear(n_hidden * 2, tag_size)

        self.dropout = nn.Dropout(dropout_p)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """
        Initialize the hidden layer.
        """
        return (Variable(torch.zeros(2, self.batch_size, self.n_hidden)),
                Variable(torch.zeros(2, self.batch_size, self.n_hidden)))

    def forward(self, sentence):
        """
        Forward pass.
        """
        embeds = self.dropout(self.embedding(sentence))
        output, (hidden, cell) = self.rnn(embeds.view(len(sentence), 1, -1))
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.hidden2label(hidden)


def predict(X: Union[str, List[torch.Tensor]], model: BidirectionalLSTM, index: Dict[int, str]) -> List[List[str]]:
    """
    Predict tag output on X.

    :param X: Can accept either a single sentence or a list of sentences.
    :param model: Model to use for prediction.
    :param index: Index of tags for interpreting output.
    :return: Tagged sentence.
    """
    predictions = []
    for line in X:
        output = model(line)
        output_arg_max = [torch.argmax(i).item() for i in output]
        sentence = [index[tag] for tag in output_arg_max]
        predictions.append(sentence)
    return predictions
