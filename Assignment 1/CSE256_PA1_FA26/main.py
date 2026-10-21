import argparse
import time
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from sentiment_data import read_word_embeddings
from DANmodels import DAN, SentimentDatasetDAN, collate_dan, train_dan
from BPEmodels import BPETokenizer, BPESentimentDataset, collate_bpe, SubwordDAN


def train_dan_with_history(model, train_loader, dev_loader, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_dev_accuracy = 0.0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = {
        "train_accuracy": [],
        "train_loss": [],
        "dev_accuracy": [],
        "dev_loss": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predictions = model(X)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()
            batch_size = y.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (predictions.argmax(dim=1) == y).sum().item()
            train_total += batch_size

        train_accuracy = train_correct / train_total
        train_loss = train_loss_sum / train_total
        model.eval()
        dev_loss_sum = 0.0
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for X, y in dev_loader:
                X = X.to(device)
                y = y.to(device)
                predictions = model(X)
                loss = loss_fn(predictions, y)
                batch_size = y.size(0)
                dev_loss_sum += loss.item() * batch_size
                dev_correct += (predictions.argmax(dim=1) == y).sum().item()
                dev_total += batch_size

        dev_accuracy = dev_correct / dev_total
        dev_loss = dev_loss_sum / dev_total
        history["train_accuracy"].append(train_accuracy)
        history["train_loss"].append(train_loss)
        history["dev_accuracy"].append(dev_accuracy)
        history["dev_loss"].append(dev_loss)

        print("Epoch {}/{} | Train loss: {:.4f} | Train accuracy: {:.4f} | Dev loss: {:.4f} | Dev accuracy: {:.4f}".format(
            epoch + 1, epochs, train_loss, train_accuracy, dev_loss, dev_accuracy
        ))

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    print("Best dev accuracy: {:.4f}".format(best_dev_accuracy))
    return best_dev_accuracy, history



def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)

    model.train()
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size

    return accuracy, average_train_loss


def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)

    model.eval()
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X = X.float()

            pred = model(X)
            loss = loss_fn(pred, y)

            eval_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size

    return accuracy, average_eval_loss


def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []

    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(
            train_loader, model, loss_fn, optimizer
        )
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(
            test_loader, model, loss_fn, optimizer
        )
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(
                "Epoch #{}: train accuracy {:.3f}, dev accuracy {:.3f}".format(
                    epoch + 1, train_accuracy, test_accuracy
                )
            )

    return all_train_accuracy, all_test_accuracy


def run_bow():
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW(
        "data/dev.txt",
        vectorizer=train_data.vectorizer,
        train=False
    )

    train_loader = DataLoader(
        train_data, batch_size=16, shuffle=True
    )
    dev_loader = DataLoader(
        dev_data, batch_size=16, shuffle=False
    )

    print(
        "Data loaded in: {:.2f} seconds".format(
            time.time() - start_time
        )
    )

    print("\n2 layers:")
    nn2_train_accuracy, nn2_dev_accuracy = experiment(
        NN2BOW(input_size=512, hidden_size=100),
        train_loader,
        dev_loader
    )

    print("\n3 layers:")
    nn3_train_accuracy, nn3_dev_accuracy = experiment(
        NN3BOW(input_size=512, hidden_size=100),
        train_loader,
        dev_loader
    )

    plt.figure(figsize=(8, 6))
    plt.plot(nn2_train_accuracy, label="2 layers")
    plt.plot(nn3_train_accuracy, label="3 layers")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy for 2 and 3 Layer Networks")
    plt.legend()
    plt.grid()
    plt.savefig("train_accuracy.png")
    plt.close()

    print("Training accuracy plot saved as train_accuracy.png")

    plt.figure(figsize=(8, 6))
    plt.plot(nn2_dev_accuracy, label="2 layers")
    plt.plot(nn3_dev_accuracy, label="3 layers")
    plt.xlabel("Epochs")
    plt.ylabel("Dev Accuracy")
    plt.title("Dev Accuracy for 2 and 3 Layer Networks")
    plt.legend()
    plt.grid()
    plt.savefig("dev_accuracy.png")
    plt.close()

    print("Dev accuracy plot saved as dev_accuracy.png")


def run_dan():
    start_time = time.time()

    embeddings_file = "data/glove.6B.50d-relativized.txt"
    word_embeddings = read_word_embeddings(embeddings_file)

    train_data = SentimentDatasetDAN(
        "data/train.txt",
        word_embeddings.word_indexer
    )

    dev_data = SentimentDatasetDAN(
        "data/dev.txt",
        word_embeddings.word_indexer
    )

    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_dan
    )

    dev_loader = DataLoader(
        dev_data,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_dan
    )

    print("\nDAN with pretrained GloVe embeddings")

    pretrained_model = DAN(
        word_embeddings,
        hidden_size=100,
        dropout=0.3,
        pretrained=True
    )

    pretrained_accuracy, pretrained_history = train_dan_with_history(
        pretrained_model,
        train_loader,
        dev_loader,
        epochs=20,
    )

    print(
        "Pretrained GloVe best dev accuracy: {:.4f}".format(
            pretrained_accuracy
        )
    )

    print("\nDAN with randomly initialized embeddings")

    random_model = DAN(
        word_embeddings,
        hidden_size=100,
        dropout=0.3,
        pretrained=False
    )

    random_accuracy, random_history = train_dan_with_history(
        random_model,
        train_loader,
        dev_loader,
        epochs=20,
    )

    print(
        "Random embeddings best dev accuracy: {:.4f}".format(
            random_accuracy
        )
    )

    plt.figure(figsize=(8, 6))
    plt.plot(pretrained_history["dev_accuracy"], label="GloVe")
    plt.plot(random_history["dev_accuracy"], label="Random")
    plt.xlabel("Epoch")
    plt.ylabel("Dev Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dan_dev_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(pretrained_history["train_loss"], label="GloVe train")
    plt.plot(pretrained_history["dev_loss"], label="GloVe dev")
    plt.plot(random_history["train_loss"], label="Random train")
    plt.plot(random_history["dev_loss"], label="Random dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dan_loss.png", dpi=300)
    plt.close()

    print("Saved dan_dev_accuracy.png and dan_loss.png")

    print("\nComparison")
    print("Pretrained GloVe: {:.4f}".format(pretrained_accuracy))
    print("Random embeddings: {:.4f}".format(random_accuracy))
    print(
        "Total DAN experiment time: {:.2f} seconds".format(
            time.time() - start_time
        )
    )


def run_subword_dan(args):
    start_time = time.time()

    print("\nSubword DAN: Byte Pair Encoding")
    print("Vocabulary sizes to test: {}".format(args.vocab_sizes))

    tokenizer = BPETokenizer()
    tokenizer.fit(
        args.train_file,
        vocab_size=max(args.vocab_sizes)
    )

    for vocab_size in args.vocab_sizes:
        experiment_start = time.time()

        print("\n" + "=" * 60)
        print("SUBWORD DAN with target BPE vocabulary size:", vocab_size)
        print("=" * 60)

        train_data = BPESentimentDataset(
            args.train_file,
            tokenizer,
            vocab_size
        )

        dev_data = BPESentimentDataset(
            args.dev_file,
            tokenizer,
            vocab_size,
            token_to_id=train_data.token_to_id
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_bpe
        )

        dev_loader = DataLoader(
            dev_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_bpe
        )

        model = SubwordDAN(
            vocab_size=len(train_data.token_to_id),
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            dropout=args.dropout
        )

        print("Actual embedding vocabulary size:", len(train_data.token_to_id))
        print("Training examples:", len(train_data))
        print("Development examples:", len(dev_data))

        best_accuracy, history = train_dan_with_history(
            model,
            train_loader,
            dev_loader,
            epochs=args.epochs,
            )

        plt.figure(figsize=(8, 6))
        plt.plot(history["dev_accuracy"], label="Dev accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SubwordDAN Dev Accuracy, BPE {}".format(vocab_size))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("bpe_{}_dev_accuracy.png".format(vocab_size), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(history["train_accuracy"], label="Train accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SubwordDAN Train Accuracy, BPE {}".format(vocab_size))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("bpe_{}_train_accuracy.png".format(vocab_size), dpi=300)
        plt.close()

        print("Saved BPE {} accuracy plots".format(vocab_size))

        elapsed = time.time() - experiment_start

        print(
            "Best dev accuracy for vocabulary size {}: {:.4f}".format(
                vocab_size, best_accuracy
            )
        )
        print(
            "Experiment time: {:.2f} seconds".format(elapsed)
        )

    print(
        "\nTotal SUBWORDDAN experiment time: {:.2f} seconds".format(
            time.time() - start_time
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train BOW, DAN, or SUBWORDDAN sentiment models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["BOW", "DAN", "SUBWORDDAN"],
        help="Model to train: BOW, DAN, or SUBWORDDAN"
    )

    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[1000, 3000, 5000],
        help="BPE vocabulary sizes to experiment with"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for DAN and SUBWORDDAN"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size for SUBWORDDAN"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Development batch size for SUBWORDDAN"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=50,
        help="Subword embedding dimension"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=100,
        help="Hidden layer size for SUBWORDDAN"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability for SUBWORDDAN"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.txt",
        help="Path to training data"
    )
    parser.add_argument(
        "--dev-file",
        type=str,
        default="data/dev.txt",
        help="Path to development data"
    )

    args = parser.parse_args()

    if args.model == "BOW":
        run_bow()
    elif args.model == "DAN":
        run_dan()
    elif args.model == "SUBWORDDAN":
        run_subword_dan(args)


if __name__ == "__main__":
    main()
