import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentiment_data import read_sentiment_examples


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer):
        self.examples = read_sentiment_examples(infile)
        self.word_indexer = word_indexer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        indices = []
        for word in example.words:
            word_index = self.word_indexer.index_of(word)

            # Use index 1 for unknown words.
            if word_index == -1:
                word_index = 1

            indices.append(word_index)

        # Handle an empty sentence.
        if len(indices) == 0:
            indices = [1]

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(example.label, dtype=torch.long)
        )


def collate_dan(batch):
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Pad sentences in a batch to the same length.
    padded_sequences = nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    return padded_sequences, labels


class DAN(nn.Module):
    def __init__(
        self,
        word_embeddings,
        hidden_size=100,
        dropout=0.3,
        pretrained=True
    ):
        super().__init__()

        self.word_indexer = word_embeddings.word_indexer
        embedding_dim = word_embeddings.get_embedding_length()
        vocab_size = len(self.word_indexer)

        if pretrained:
            # Initialize from GloVe and allow fine-tuning.
            self.embedding = (
                word_embeddings.get_initialized_embedding_layer(
                    frozen=False
                )
            )
        else:
            # Randomly initialize embeddings.
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0
            )

            nn.init.normal_(
                self.embedding.weight,
                mean=0.0,
                std=0.1
            )

            # Keep the padding vector at zero.
            with torch.no_grad():
                self.embedding.weight[0].zero_()

        self.hidden = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch_size, sentence_length)
        embeddings = self.embedding(x)

        # Ignore PAD tokens when averaging.
        mask = (x != 0).unsqueeze(-1).float()
        summed_embeddings = (embeddings * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1.0)
        averaged_embeddings = summed_embeddings / lengths

        hidden = F.relu(self.hidden(averaged_embeddings))
        hidden = self.dropout(hidden)

        logits = self.output(hidden)
        return F.log_softmax(logits, dim=1)


def evaluate_dan(model, data_loader, loss_fn, device):
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


def train_dan(model, train_loader, dev_loader, epochs=20, lr=0.001):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    best_dev_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
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
            total_loss += loss.item() * batch_size
            correct += (predictions.argmax(dim=1) == y).sum().item()
            total += batch_size

        train_accuracy = correct / total
        train_loss = total_loss / total

        dev_accuracy, dev_loss = evaluate_dan(
            model, dev_loader, loss_fn, device
        )

        print(
            "Epoch {}/{} | Train loss: {:.4f} | Train accuracy: {:.4f} "
            "| Dev loss: {:.4f} | Dev accuracy: {:.4f}".format(
                epoch + 1,
                epochs,
                train_loss,
                train_accuracy,
                dev_loss,
                dev_accuracy
            )
        )

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print("Best dev accuracy: {:.4f}".format(best_dev_accuracy))

    return best_dev_accuracy
