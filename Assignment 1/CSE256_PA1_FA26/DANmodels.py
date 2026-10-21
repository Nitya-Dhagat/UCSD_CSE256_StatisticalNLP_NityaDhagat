import copy
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples
from utils import Indexer


def set_seed(seed):
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_indexer(train_file):
    """
    Word indexer built ONLY from the training file (PAD=0, UNK=1).
    Used for a fair random-embedding baseline: words that never appear in
    training map to UNK at dev time instead of keeping an untrained random vector.
    """
    indexer = Indexer()
    indexer.add_and_get_index("PAD")
    indexer.add_and_get_index("UNK")
    for ex in read_sentiment_examples(train_file):
        for word in ex.words:
            indexer.add_and_get_index(word)
    return indexer


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer):
        self.examples = read_sentiment_examples(infile)
        self.word_indexer = word_indexer
        self.encoded = []

        unk_count = 0
        token_count = 0
        for example in self.examples:
            indices = []
            for word in example.words:
                word_index = word_indexer.index_of(word)
                if word_index == -1:
                    word_index = 1  # UNK
                    unk_count += 1
                indices.append(word_index)
            token_count += len(indices)
            if len(indices) == 0:
                indices = [1]
            self.encoded.append(indices)

        self.unk_rate = unk_count / max(token_count, 1)
        self.avg_length = token_count / max(len(self.examples), 1)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.examples[idx].label, dtype=torch.long)
        )


def collate_dan(batch):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    padded_sequences = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    return padded_sequences, labels


class DAN(nn.Module):
    """
    Deep Averaging Network.

    word_embeddings given  -> embedding layer initialised from GloVe
                              (trainable unless freeze_embeddings=True)
    word_embeddings None   -> random N(0, 0.1) embeddings of size
                              (vocab_size, embedding_dim), trained from scratch
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        word_embeddings=None,
        freeze_embeddings=False,
        hidden_size=100,
        num_hidden_layers=1,
        dropout=0.3,
        embedding_dropout=0.0,
        activation="relu"
    ):
        super().__init__()

        if word_embeddings is not None:
            self.embedding = word_embeddings.get_initialized_embedding_layer(
                frozen=freeze_embeddings
            )
            embedding_dim = word_embeddings.get_embedding_length()
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            with torch.no_grad():
                self.embedding.weight[0].zero_()
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # Dropout applied to the averaged sentence vector (after the embedding layer).
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError("activation must be one of {}".format(list(activations)))

        layers = []
        in_dim = embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(activations[activation]())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 2)

    def forward(self, x):
        # x: (batch, sentence_length), PAD = 0
        embeddings = self.embedding(x)

        # Ignore PAD tokens when averaging.
        mask = (x != 0).unsqueeze(-1).float()
        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged = summed / lengths

        averaged = self.embedding_dropout(averaged)
        hidden = self.hidden(averaged)
        return F.log_softmax(self.output(hidden), dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            predictions = model(X)
            loss = loss_fn(predictions, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions.argmax(dim=1) == y).sum().item()
            total += batch_size
    return correct / total, total_loss / total


def train_model(
    model,
    train_loader,
    dev_loader,
    epochs=20,
    lr=0.001,
    weight_decay=1e-5,
    optimizer_name="adam",
    target_dev_accuracy=0.77,
    verbose=True
):
    """
    Single training routine shared by DAN and SubwordDAN.

    Loss: NLLLoss on log-softmax outputs. The model with the best dev accuracy
    (first epoch reaching the max) is restored at the end.

    Returns a dict with history and convergence statistics. Epochs are 1-indexed.
    train_time covers training + per-epoch dev evaluation (no data loading).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.NLLLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    history = {
        "train_accuracy": [], "train_loss": [],
        "dev_accuracy": [], "dev_loss": []
    }
    best_dev_accuracy = 0.0
    best_epoch = 0
    time_to_best = 0.0
    best_state = copy.deepcopy(model.state_dict())

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(X)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            loss_sum += loss.item() * batch_size
            correct += (predictions.argmax(dim=1) == y).sum().item()
            total += batch_size

        train_accuracy = correct / total
        train_loss = loss_sum / total
        dev_accuracy, dev_loss = evaluate_model(model, dev_loader, loss_fn, device)

        history["train_accuracy"].append(train_accuracy)
        history["train_loss"].append(train_loss)
        history["dev_accuracy"].append(dev_accuracy)
        history["dev_loss"].append(dev_loss)

        if verbose:
            print(
                "Epoch {}/{} | Train loss: {:.4f} | Train accuracy: {:.4f} "
                "| Dev loss: {:.4f} | Dev accuracy: {:.4f}".format(
                    epoch, epochs, train_loss, train_accuracy, dev_loss, dev_accuracy
                )
            )

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch
            time_to_best = time.time() - start
            best_state = copy.deepcopy(model.state_dict())

    train_time = time.time() - start
    model.load_state_dict(best_state)

    epochs_to_target = None
    for i, acc in enumerate(history["dev_accuracy"]):
        if acc >= target_dev_accuracy:
            epochs_to_target = i + 1
            break

    dev_losses = history["dev_loss"]
    min_dev_loss_epoch = int(np.argmin(dev_losses)) + 1

    return {
        "history": history,
        "best_dev_accuracy": best_dev_accuracy,
        "best_epoch": best_epoch,
        "train_accuracy_at_best": history["train_accuracy"][best_epoch - 1],
        "final_dev_accuracy": history["dev_accuracy"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "epochs_to_target": epochs_to_target,
        "min_dev_loss": float(min(dev_losses)),
        "min_dev_loss_epoch": min_dev_loss_epoch,
        "train_time": train_time,
        "time_to_best": time_to_best,
        "num_params": count_parameters(model),
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }