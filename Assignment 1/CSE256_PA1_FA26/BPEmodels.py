import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from sentiment_data import read_sentiment_examples


END_OF_WORD = "</w>"


def _initial_symbols(word):
    """Represent a word as characters, marking its final character."""
    if not word:
        return tuple()
    return tuple(list(word[:-1]) + [word[-1] + END_OF_WORD])


def _merge_pair(symbols, pair):
    """Merge every non-overlapping occurrence of pair, left to right."""
    if len(symbols) < 2:
        return symbols

    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    """
    Frequency-weighted BPE trained only on the training split.

    vocab_size is the target size of the BPE symbol inventory. PAD and UNK
    are added separately by the downstream sentiment vocabulary.
    """

    def __init__(self):
        self.merges = []
        self.base_vocab_size = 0
        self.fitted = False
        self._token_cache = {}

    def fit(self, train_file, vocab_size=5000):
        examples = read_sentiment_examples(train_file)
        word_counts = Counter(
            word for example in examples for word in example.words if word
        )
        if not word_counts:
            raise ValueError("No words were found in the BPE training file.")

        word_symbols = {
            word: _initial_symbols(word) for word in word_counts
        }
        base_symbols = {
            symbol
            for symbols in word_symbols.values()
            for symbol in symbols
        }
        self.base_vocab_size = len(base_symbols)
        target_merges = max(0, vocab_size - self.base_vocab_size)

        print("Unique training words:", len(word_counts))
        print("Initial BPE symbols:", self.base_vocab_size)
        print("Target BPE vocabulary size:", vocab_size)
        print("Learning up to", target_merges, "merge rules...")

        # Build pair counts and an index of which word types contain each pair.
        # Updating only affected word types is much faster than rescanning the
        # entire vocabulary after every merge.
        pair_counts = Counter()
        pair_to_words = {}
        for word, symbols in word_symbols.items():
            frequency = word_counts[word]
            local_counts = Counter(zip(symbols, symbols[1:]))
            for pair, count in local_counts.items():
                pair_counts[pair] += frequency * count
                pair_to_words.setdefault(pair, set()).add(word)

        self.merges = []
        self._token_cache = {}
        start = time.time()

        for merge_number in range(target_merges):
            if not pair_counts:
                break

            # Counter preserves first-seen order for frequency ties.
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 0:
                break
            self.merges.append(best_pair)

            affected_words = list(pair_to_words.get(best_pair, set()))
            for word in affected_words:
                old_symbols = word_symbols[word]
                new_symbols = _merge_pair(old_symbols, best_pair)
                frequency = word_counts[word]

                old_counts = Counter(zip(old_symbols, old_symbols[1:]))
                new_counts = Counter(zip(new_symbols, new_symbols[1:]))

                # Remove the word's old pair contributions.
                for pair, count in old_counts.items():
                    pair_counts[pair] -= frequency * count

                # Update the reverse index for pairs no longer/newly present.
                for pair in old_counts.keys() - new_counts.keys():
                    pair_to_words[pair].discard(word)
                for pair in new_counts.keys() - old_counts.keys():
                    pair_to_words.setdefault(pair, set()).add(word)

                # Add the word's new pair contributions.
                for pair, count in new_counts.items():
                    pair_counts[pair] += frequency * count

                word_symbols[word] = new_symbols

            # Remove exhausted pairs so max() only considers valid candidates.
            for pair in list(pair_counts.keys()):
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)

            if (merge_number + 1) % 500 == 0:
                print(
                    "  Learned {}/{} merges ({:.1f}s)".format(
                        merge_number + 1,
                        target_merges,
                        time.time() - start
                    )
                )

        self.fitted = True
        print(
            "BPE training finished: {} merges in {:.2f}s".format(
                len(self.merges), time.time() - start
            )
        )
        return self

    def tokenize_word(self, word, vocab_size=5000):
        if not self.fitted:
            raise RuntimeError("Call fit() before tokenizing.")

        number_of_merges = min(
            len(self.merges),
            max(0, vocab_size - self.base_vocab_size)
        )
        cache_key = (word, number_of_merges)
        if cache_key in self._token_cache:
            return list(self._token_cache[cache_key])

        symbols = _initial_symbols(word)
        for pair in self.merges[:number_of_merges]:
            symbols = _merge_pair(symbols, pair)

        # Remove the end-of-word marker from the final piece.
        pieces = []
        for symbol in symbols:
            if symbol.endswith(END_OF_WORD):
                symbol = symbol[:-len(END_OF_WORD)]
            if symbol:
                pieces.append(symbol)

        self._token_cache[cache_key] = tuple(pieces)
        return pieces

    def tokenize_sentence(self, words, vocab_size=5000):
        pieces = []
        for word in words:
            pieces.extend(self.tokenize_word(word, vocab_size))
        return pieces


class BPESentimentDataset(Dataset):
    """Sentiment examples represented as BPE subword IDs."""

    def __init__(self, infile, tokenizer, vocab_size, token_to_id=None):
        self.examples = read_sentiment_examples(infile)
        self.tokenizer = tokenizer
        self.bpe_vocab_size = vocab_size

        # Encode once rather than re-tokenizing every training epoch.
        self.tokenized = [
            tokenizer.tokenize_sentence(example.words, vocab_size)
            for example in self.examples
        ]

        if token_to_id is None:
            self.token_to_id = {"PAD": 0, "UNK": 1}
            for pieces in self.tokenized:
                for piece in pieces:
                    if piece not in self.token_to_id:
                        self.token_to_id[piece] = len(self.token_to_id)
        else:
            self.token_to_id = token_to_id

        self.encoded = []
        for pieces in self.tokenized:
            ids = [self.token_to_id.get(piece, 1) for piece in pieces]
            if not ids:
                ids = [1]
            self.encoded.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.examples[idx].label, dtype=torch.long)
        )

    def average_tokens_per_sentence(self):
        if not self.tokenized:
            return 0.0
        return sum(len(x) for x in self.tokenized) / len(self.tokenized)


def collate_bpe(batch):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    padded = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    return padded, labels


class SubwordDAN(nn.Module):
    """DAN with randomly initialized, trainable subword embeddings."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_size=100,
        dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        embeddings = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / lengths
        hidden = F.relu(self.hidden(averaged))
        hidden = self.dropout(hidden)
        return F.log_softmax(self.output(hidden), dim=1)
