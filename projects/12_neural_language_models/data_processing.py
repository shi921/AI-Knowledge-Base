import os
import pickle as pkl

import numpy as np
import torch
import torchtext
from torch.utils.data import TensorDataset

token_start = "<sos>"
token_end = "<eos>"
token_unk = "<unk>"
token_pad = "<pad>"

max_seq_len = 500

data_dir = os.path.join(os.path.dirname(__file__), "data")


data_dir_wikitext_103 = os.path.join(data_dir, "wikitext-103")
data_dir_wikitext_2 = os.path.join(data_dir, "wikitext-2")


def create_vocabulary(tokens, default_token='<unk>', pad_token=None):
    vocab = torchtext.vocab.build_vocab_from_iterator(tokens, min_freq=3)
    if not default_token in vocab:
        vocab.append_token(default_token)
    vocab.set_default_index(vocab[default_token])

    if pad_token:
        vocab.append_token(pad_token)

    return vocab


def load_and_tokenize(data_path, tokenizer):
    with open(data_path, "r", encoding="utf-8") as file:
        text_sentences = file.read().split("\n")

    text_sentences = [sentence.strip() for sentence in text_sentences if
                      len(sentence.strip()) > 1 and not sentence.strip().startswith("=")]
    tokens = [tokenizer("<sos> " + sentence + " <eos>") for sentence in text_sentences]

    return tokens


def encode_tokens(tokens, vocab, pad_length=None, pad_token_index=None):
    # Convert tokens to indices using the vocabulary
    tokens_encoded = [vocab(sentence) for sentence in tokens]

    # Determine the maximum sequence length
    if pad_length is None:
        pad_length = max(len(sentence) for sentence in tokens_encoded)

    # Initialize a NumPy array with the padding token index
    tokens_padded = np.full((len(tokens_encoded), pad_length), pad_token_index, dtype=np.int64)

    # Populate the array with encoded tokens, truncating if necessary
    for i, sentence in enumerate(tokens_encoded):
        tokens_padded[i, :min(len(sentence), pad_length)] = sentence[:pad_length]

    return tokens_padded


def build_tensor_dataset_from_tokens(tokens, vocab, max_seq_length, pad_token_index):
    tokens_encoded = encode_tokens(tokens, vocab, max_seq_length, pad_token_index)
    tokens_encoded_tensor = torch.tensor(np.stack(tokens_encoded), dtype=torch.long)
    dataset = TensorDataset(tokens_encoded_tensor[:, :-1], tokens_encoded_tensor[:, 1:])

    return dataset


def load_dataset_from_file(file_path, tokenizer, vocab, max_seq_length, pad_token_index):
    tokens = load_and_tokenize(file_path, tokenizer)
    dataset = build_tensor_dataset_from_tokens(tokens, vocab, max_seq_length, pad_token_index)
    return dataset


def load_data(train_data_path, valid_data_path, test_data_path):
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    tokens_train = load_and_tokenize(train_data_path, tokenizer)

    vocab = create_vocabulary(tokens_train, default_token=token_unk, pad_token=token_pad)

    max_seq_length = min(max([len(tokens) for tokens in tokens_train]), max_seq_len)
    pad_token_index = vocab[token_pad]

    train_data = build_tensor_dataset_from_tokens(tokens_train, vocab, max_seq_length, pad_token_index)

    if valid_data_path:
        valid_data = load_dataset_from_file(valid_data_path, tokenizer, vocab, max_seq_length, pad_token_index)
    else:
        valid_data = None

    if test_data_path:
        test_data = load_dataset_from_file(test_data_path, tokenizer, vocab, max_seq_length, pad_token_index)
    else:
        test_data = None

    return train_data, valid_data, test_data, vocab


def store_vocab(vocab, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(vocab, f)


def load_vocab(file_path):
    with open(file_path, 'rb') as f:
        vocab = pkl.load(f)
    return vocab


def load_wikitext_103():
    return load_data(os.path.join(data_dir, "wikitext-103/wiki.train.tokens"),
                     os.path.join(data_dir, "wikitext-103/wiki.valid.tokens"),
                     os.path.join(data_dir, "wikitext-103/wiki.test.tokens"))


if __name__ == '__main__':
    data_train, data_valid, data_test, vocab = load_wikitext_103()
    print(len(vocab))
