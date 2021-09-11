import random

import numpy as np
import torch

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


def create_text_from_label_mnist(len_seq, label, alphabet):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot


def seq2text(alphabet, seq):
    return [alphabet[seq[j]] for j in range(len(seq))]


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples;
