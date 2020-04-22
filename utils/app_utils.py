import torch

from typing import List, Dict
from utils.data_process import flatten, prepare_sentence
from model.model import BidirectionalLSTM, predict


def load_model(path: str, token_index: Dict[str, int], tag_index: Dict[int, str]) -> BidirectionalLSTM:
    """
    Load the pretrained model.

    :param path: path to trained model
    :param token_index: token index
    :param tag_index: tag index
    :return: loaded model
    """
    batch_size = 32
    vocab_size = len(token_index)
    tag_size = len(tag_index)
    embedding_dim = 128
    n_hidden = 256
    dropout = 0.5

    model = BidirectionalLSTM(vocab_size, tag_size, embedding_dim, n_hidden, batch_size, dropout)
    model.load_state_dict(torch.load(path))
    return model


def predict_on_sentence(sentence: str,
                        model: BidirectionalLSTM,
                        token_index: Dict[str, int],
                        tag_index: Dict[int, str]) -> List[List]:
    """
    Predict on a single sentence using the model.

    :param sentence: a string.
    :param model: Model to use for prediction.
    :param token_index: Index of tokens, used to prepare the sentence for prediction.
    :param tag_index: Index of tokens, to make output interpretable.
    :return: A list with the token and its predicted tab.
    """
    if sentence[-1] == ".":
        sentence = sentence[:-1]

    sentence = sentence.split()
    prepared = prepare_sentence(sentence, token_index)
    pred = predict([prepared], model, tag_index)
    return [[w, t] for w, t in zip(sentence, flatten(pred))]
