import torch
import json

from torch.nn.utils.rnn import pad_sequence
from typing import List, Any, Dict, Generator


def parse_file(filename: str) -> (List[List[str]], List[List[str]]):
    """
    Function for parsing a file in the CONLL format.

    :param filename: Path to text file. needs to be in CONLL format.
    :return: List of lists of tokens, list of lists of tags from file.
    """
    tokens = []
    tags = []

    with open(filename) as f:
        sentence = []
        sentence_tags = []
        for line in f:
            line_split = line.split()
            if len(line_split) == 0:
                if len(sentence) > 1:
                    tokens.append(sentence)
                    tags.append(sentence_tags)
                    sentence = []
                    sentence_tags = []
            else:
                if line_split[0] != "-DOCSTART-":
                    sentence.append(line_split[0])
                    sentence_tags.append(line_split[-1])
                else:
                    continue

    return tokens[1:], tags[1:]


def flatten(list_of_lists: List[List[Any]], unique: bool = False) -> List[Any]:
    """
    Flattens and can reduce a list of lists to unique values

    :param list_of_lists: list of lists to flatten
    :param unique: specifying whether to return a unique or non-unique list
    :return: either a unique or non-unique list
    """
    flattened = [item for sublist in list_of_lists for item in sublist]
    if unique:
        return list(set(flattened))
    else:
        return flattened


def index_dicts(list_to_index: List[List[Any]], special_tokens: List = None) -> Dict[str, int]:
    """
    Creates the index of tokens or tags to be used in model

    :param list_to_index: list to index
    :param special_tokens: special tokens, if any, for index
    :return: dictionary with indices
    """
    word2idx = {}
    flat_list = flatten(list_to_index, unique=True)
    if special_tokens:
        flat_list = [i for i in flat_list if i not in special_tokens]
        flat_list = special_tokens + flat_list

    for v, k in enumerate(flat_list):
        if k not in word2idx.keys():
            word2idx[k] = v

    return word2idx


def prepare_sentence(sentence: List[str], index: Dict[str, int]):
    """
    Transforms a sentence into a tensor.

    :param sentence: A single sentence, in list format.
    :param index: The dictionary containing the index of words.
    :return: A tensor containing the integer index for each sentence.
    """
    idxs = [index[w] if w in index.keys() else index["<UNK>"] for w in sentence]
    return torch.tensor(idxs, dtype=torch.long)


def convert_to_index(sequences: [List[List[Any]]], index: Dict[str, int]) -> List[torch.Tensor]:
    """
    Transforms a list of sentences into tensors.

    :param sequences: List of sentences to convert to tensor form.
    :param index: The dictionary containing the index of words.
    :return: A list of tensors for each sentence.
    """
    converted = []
    for sentence in sequences:
        converted.append(prepare_sentence(sentence, index))
    return converted


def generate_batches(batch_size: int, tokens: List[torch.Tensor], tags: List[torch.Tensor]) -> (Generator, Generator):
    """
    Generates batches for model training.

    :param batch_size: Size of the batch (int).
    :param tokens: A list of tensors that have been converted from words to integer indices.
    :param tags: Corresponding tags for the batch.
    :return: Yields a token and tag batch.
    """

    for i in range(0, len(tokens), batch_size):
        token_chunk = tokens[i:i + batch_size]
        tag_chunk = tags[i:i + batch_size]

        tokens_padded = pad_sequence([word for word in token_chunk], batch_first=True)
        tags_padded = pad_sequence([tag for tag in tag_chunk], batch_first=True)
        yield tokens_padded, tags_padded


def dict_to_file(index: Dict[Any, Any], filepath: str):
    """
    Convert a dictionary into a json file.

    :param index: Dict to be converted.
    :param filepath: Name of file.
    """
    with open(filepath, "w") as f:
        json.dump(index, f)

    print("Done.")


def file_to_dict(filepath: str, int_key: bool = False) -> Dict[Any, Any]:
    """
    Convert json file into dictionary.

    :param filepath: Name of file to be converted.
    :param int_key: If the dict keys are integers, return them as such.
    :return: Dictionary!
    """
    with open(filepath) as f:
        if int_key is True:
            loaded = json.load(f)
            return {int(key): val for key, val in loaded.items()}
        else:
            return json.load(f)
