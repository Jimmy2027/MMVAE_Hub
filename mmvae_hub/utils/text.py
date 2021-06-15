import numpy as np
import torch
from typing import List, Iterable, Union

digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun']
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq: int, alphabet: str, seq: str) -> torch.tensor:
    """
    One hot encodes the sequence.
    Set $ for the end of text. Pads with & to len_seq. Replaces chars that are not found in the alphabet with @.
    len_seq is the maximum sequence length

    """
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq]
    elif len(seq) < len_seq:
        seq += '$'
        seq = seq.ljust(len_seq, '&')

    for index_char, char in enumerate(seq):
        # char2Index return -1 if the char is not found in the alphabet.
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
        else:
            X[index_char, alphabet.find('@')] = 1.0

    return X


